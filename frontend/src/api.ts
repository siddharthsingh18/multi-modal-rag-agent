export type Source = {
  id: string;
  score: number;
  content: string;
};

export type QueryResult = {
  query: string;
  answer: string;
  documents: Source[];
  reflection?: { score?: number };
  metadata?: { num_documents?: number; iterations?: number };
};

const baseUrl = import.meta.env.VITE_API_BASE_URL || "";

async function request<T>(path: string, options: RequestInit = {}, apiKey = "") {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
      ...options.headers,
    },
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail?.message || body.detail || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function ingestText(text: string, apiKey: string) {
  return request<{ success: boolean; num_chunks: number; message: string }>(
    "/api/v1/ingest",
    {
      method: "POST",
      body: JSON.stringify({ text, metadata: { source: "workspace", type: "text" } }),
    },
    apiKey,
  );
}

export function ingestFile(file: File, apiKey: string) {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${baseUrl}/api/v1/ingest/upload`, {
    method: "POST",
    headers: apiKey ? { "X-API-Key": apiKey } : {},
    body: form,
  }).then(async (response) => {
    if (!response.ok) throw new Error((await response.json()).detail || "Upload failed");
    return response.json() as Promise<{ success: boolean; num_chunks: number; message: string }>;
  });
}

export function queryRag(query: string, apiKey: string) {
  return request<QueryResult>(
    "/api/v1/query",
    { method: "POST", body: JSON.stringify({ query, top_k: 5, use_reflection: true }) },
    apiKey,
  );
}

export function getHealth() {
  return request<{ status: string; dependencies: Record<string, string> }>("/api/v1/health");
}
