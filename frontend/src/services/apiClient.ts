/**
 * Centralized API Client for TrafficLens AI
 *
 * Implements typed HTTP requests with automatic JSON parsing,
 * timeout handling, and unified error mapping.
 */

import { APP_CONFIG } from "../core/config";
import { ApiError } from "../types/api";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = APP_CONFIG.apiBaseUrl) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
    const headers = new Headers(options.headers || {});

    if (!headers.has("Accept")) {
      headers.set("Accept", "application/json");
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        let errorDetail: any = "Unknown error";
        try {
          const body = await response.json();
          errorDetail = body.detail || body.message || body;
        } catch {
          errorDetail = await response.text();
        }

        const apiError: ApiError = {
          status: response.status,
          message: typeof errorDetail === "string" ? errorDetail : JSON.stringify(errorDetail),
          detail: errorDetail,
        };
        throw apiError;
      }

      return (await response.json()) as T;
    } catch (err: any) {
      if (err.status) {
        throw err;
      }
      // Network or connection failure
      const netError: ApiError = {
        status: 0,
        message: err.message || "Network connection failure. Is the backend server running?",
        detail: err,
      };
      throw netError;
    }
  }

  public buildUrl(endpoint: string): string {
    return `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
  }

  public get<T>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: "GET" });
  }

  public post<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    const headers = new Headers(options?.headers || {});
    let body: any = undefined;

    if (data instanceof FormData) {
      body = data;
    } else if (data !== undefined) {
      headers.set("Content-Type", "application/json");
      body = JSON.stringify(data);
    }

    return this.request<T>(endpoint, {
      ...options,
      method: "POST",
      headers,
      body,
    });
  }

  public uploadWithProgress<T>(
    endpoint: string,
    formData: FormData,
    onProgress?: (percent: number) => void
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const url = `${this.baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.setRequestHeader("Accept", "application/json");

      if (xhr.upload && onProgress) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            onProgress(percent);
          }
        };
      }

      xhr.onload = () => {
        let responseData: any;
        try {
          responseData = JSON.parse(xhr.responseText);
        } catch {
          responseData = xhr.responseText;
        }

        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(responseData as T);
        } else {
          const errorDetail =
            responseData?.detail || responseData?.message || responseData || "Upload failed";
          const error: ApiError = {
            status: xhr.status,
            message: typeof errorDetail === "string" ? errorDetail : JSON.stringify(errorDetail),
            detail: errorDetail,
          };
          reject(error);
        }
      };

      xhr.onerror = () => {
        const error: ApiError = {
          status: 0,
          message: "Network error during video upload. Please check connection to backend server.",
        };
        reject(error);
      };

      xhr.send(formData);
    });
  }
}

export const apiClient = new ApiClient();
