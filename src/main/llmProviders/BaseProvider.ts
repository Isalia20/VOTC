import {
  ILLMCompletionRequest,
  ILLMModel,
  ILLMOutput,
  LLMProviderConfig
} from './types';

/**
 * Base implementation for LLM providers with common functionality
 */
export abstract class BaseProvider {
  abstract readonly providerId: string;
  abstract readonly name: string;

  /**
   * Validates the provider configuration
   * @param config - The provider configuration to validate
   */
  protected validateConfig(config: LLMProviderConfig): void {
    if (!config.apiKey) {
      throw new Error(`API key is required for ${this.name}`);
    }

    if (this.requiresBaseUrl && !config.baseUrl) {
      throw new Error(`Base URL is required for ${this.name}`);
    }

    if (!config.defaultModel) {
      throw new Error(`Default model is required for ${this.name}`);
    }
  }

  /**
   * Whether this provider requires a base URL
   * Override in subclasses if needed
   */
  protected requiresBaseUrl = false;

  /**
   * Tests connection to the LLM provider
   * @param config - Provider configuration
   * @returns Connection test result
   */
  async testConnection(config: LLMProviderConfig): Promise<{success: boolean, error?: string, message?: string}> {
    try {
      this.validateConfig(config);

      const testMessages = [{
        role: 'user' as const,
        content: 'Hi.'
      }];

      // Send a minimal completion request
      await this.chatCompletion({
        model: config.defaultModel!,
        messages: testMessages,
        max_tokens: 10,
        stream: false
      }, config);

      return {
        success: true,
        message: `Successfully connected to ${this.name}`
      };
    } catch (error: any) {
      console.error(`[${this.providerId}] Connection test failed:`, error);
      return {
        success: false,
        error: error.message || `Failed to connect to ${this.name}`
      };
    }
  }

  /**
   * Lists available models (if provider supports it)
   * @param config - Provider configuration
   * @returns Array of available models
   */
  async listModels(config: LLMProviderConfig): Promise<ILLMModel[]> {
    try {
      this.validateConfig(config);

      // Default implementation - providers should override if they support model listing
      console.warn(`[${this.providerId}] Model listing not implemented for ${this.name}`);
      return [];
    } catch (error: any) {
      console.error(`[${this.providerId}] Error listing models:`, error);
      throw error;
    }
  }

  /**
   * Send chat completion request
   * @param request - The completion request
   * @param config - Provider configuration
   * @returns Completion response or stream
   */
  abstract chatCompletion(
    request: ILLMCompletionRequest,
    config: LLMProviderConfig
  ): ILLMOutput;

  /**
   * Helper method to create consistent error messages
   */
  protected createErrorMessage(operation: string, error: any): string {
    return `[${this.providerId}] ${operation} failed: ${error.message || 'Unknown error'}`;
  }

  /**
   * Helper method to validate messages format
   */
  protected validateMessages(messages: ILLMCompletionRequest['messages']): void {
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new Error('Messages must be a non-empty array');
    }

    for (const message of messages) {
      if (!message.role || !message.content) {
        throw new Error('Each message must have role and content');
      }
    }
  }

  protected getAPIKey(config: LLMProviderConfig): string {
    if (!config.apiKey) {
      throw new Error(`Invalid configuration for ${this.name}: API key is missing.`);
    }
    return config.apiKey;
  }

  protected getBaseUrl(config: LLMProviderConfig): string {
    if (!config.baseUrl) {
      throw new Error(`Invalid configuration for ${this.name}: Base URL is missing.`);
    }
    return config.baseUrl;
  }

  /**
   * Retry an async operation with exponential backoff
   * @param operation The async operation to retry
   * @param maxRetries Maximum number of retry attempts (default: 3)
   * @param initialDelay Initial delay in milliseconds (default: 1000)
   * @param shouldRetry Function to determine if error is retryable (default: always retry)
   * @returns Promise resolving to operation result
   */
  /**
   * Transform a json_schema response_format into json_object + prompt injection.
   * This is needed for providers that don't natively support json_schema
   * (e.g., DeepSeek, Claude via OpenRouter/OpenAI-compatible, Kimi, etc.)
   *
   * The schema description is injected into the last system message so the model
   * follows the expected structure via instruction-following.
   */
  protected transformJsonSchemaToPromptInjection(request: ILLMCompletionRequest): ILLMCompletionRequest {
    if (request.response_format?.type !== 'json_schema' || !request.response_format.json_schema) {
      return request;
    }

    const transformed = { ...request };
    const jsonSchemaObj = request.response_format.json_schema;
    const schemaName = jsonSchemaObj.name || 'response';
    const schemaObj = jsonSchemaObj.schema;

    // Convert json_schema to json_object and inject schema into the system prompt
    transformed.response_format = { type: 'json_object' };

    // Build human-readable schema description
    const schemaDescription = this.buildSchemaDescription(schemaName, schemaObj);

    // Find the last system message and append schema info, or add a new system message
    const messages = [...request.messages];
    let schemaInjected = false;

    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'system') {
        messages[i] = {
          ...messages[i],
          content: messages[i].content + '\n\n' + schemaDescription
        };
        schemaInjected = true;
        break;
      }
    }

    if (!schemaInjected) {
      messages.unshift({
        role: 'system',
        content: schemaDescription
      });
    }

    transformed.messages = messages;
    return transformed;
  }

  /**
   * Build a human-readable schema description for prompt injection
   */
  private buildSchemaDescription(schemaName: string, schema: any): string {
    let description = `You MUST respond with valid JSON matching this schema:\n`;
    description += `Schema name: ${schemaName}\n`;

    if (schema.properties) {
      description += this.describeObjectSchema(schema, 0);
    }

    description += `\nIMPORTANT: Your response must be ONLY valid JSON. No prose, no code fences, no explanations.`;

    return description;
  }

  /**
   * Recursively describe an object schema
   */
  private describeObjectSchema(schema: any, indent: number): string {
    const spaces = '  '.repeat(indent);
    let result = '';

    if (schema.type === 'object' && schema.properties) {
      const required = schema.required || [];

      for (const [key, value] of Object.entries(schema.properties)) {
        const isRequired = required.includes(key);
        const reqMarker = isRequired ? ' (required)' : ' (optional)';

        if ((value as any).type === 'array') {
          const items = (value as any).items;
          if ((items as any).anyOf) {
            result += `${spaces}- ${key}: array of objects${reqMarker}\n`;
            result += this.describeAnyOfSchema(items, indent + 1);
          } else if ((items as any).type === 'object') {
            result += `${spaces}- ${key}: array of objects${reqMarker}\n`;
            result += this.describeObjectSchema(items, indent + 1);
          } else {
            result += `${spaces}- ${key}: array of ${(items as any).type}${reqMarker}\n`;
          }
        } else if ((value as any).type === 'object') {
          result += `${spaces}- ${key}: object${reqMarker}\n`;
          result += this.describeObjectSchema(value, indent + 1);
        } else if ((value as any).anyOf) {
          result += `${spaces}- ${key}: ${(value as any).anyOf.map((t: any) => t.type).join(' | ')}${reqMarker}\n`;
        } else if ((value as any).const !== undefined) {
          result += `${spaces}- ${key}: "${(value as any).const}" (constant)${reqMarker}\n`;
        } else if ((value as any).enum) {
          result += `${spaces}- ${key}: enum{${(value as any).enum.join(', ')}}${reqMarker}\n`;
        } else {
          result += `${spaces}- ${key}: ${(value as any).type}${reqMarker}\n`;
        }
      }
    }

    return result;
  }

  /**
   * Describe an anyOf schema (used for action variants)
   */
  private describeAnyOfSchema(schema: any, indent: number): string {
    const spaces = '  '.repeat(indent);
    let result = '';

    if (schema.anyOf) {
      schema.anyOf.forEach((variant: any, index: number) => {
        if (variant.properties?.actionId?.const) {
          const actionId = variant.properties.actionId.const;
          result += `${spaces}Variant ${index + 1} (actionId: "${actionId}"):\n`;
          result += this.describeObjectSchema(variant, indent + 1);
        }
      });
    }

    return result;
  }

  protected async retryWithBackoff<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    initialDelay: number = 1000,
    shouldRetry: (error: any) => boolean = () => true
  ): Promise<T> {
    let delay = initialDelay;
    let lastError: any;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        if (attempt === maxRetries || !shouldRetry(error)) {
          throw error;
        }
        console.log(`[${this.providerId}] Retry attempt ${attempt + 1}/${maxRetries + 1} after ${delay}ms delay`);
        await new Promise(resolve => setTimeout(resolve, delay));
        delay *= 2; // Exponential backoff
      }
    }
    throw lastError;
  }
}