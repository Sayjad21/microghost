import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: NextRequest) {
  const spaceUrl = process.env.HF_SPACE_URL;
  if (!spaceUrl) {
    return NextResponse.json(
      {
        ok: false,
        error: "HF_SPACE_URL is not configured. Set it to your Hugging Face Space URL."
      },
      { status: 500 }
    );
  }

  const formData = await request.formData();
  const upstreamUrl = `${spaceUrl.replace(/\/$/, "")}/analyze`;

  const headers: HeadersInit = {};
  if (process.env.HF_TOKEN) {
    headers.Authorization = `Bearer ${process.env.HF_TOKEN}`;
  }

  try {
    const response = await fetch(upstreamUrl, {
      method: "POST",
      headers,
      body: formData
    });

    const text = await response.text();
    let payload: unknown;
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { ok: false, error: text || "Empty response from inference backend." };
    }

    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Could not reach inference backend."
      },
      { status: 502 }
    );
  }
}
