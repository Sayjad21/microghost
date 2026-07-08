import { Client, handle_file } from "@gradio/client";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

function normalizeSpaceTarget(spaceUrl: string) {
  return spaceUrl.replace(/\/$/, "");
}

function firstResult(data: unknown) {
  if (Array.isArray(data)) return data[0];
  return data;
}

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
  const rgbFile = formData.get("rgb_image");
  const thermalFile = formData.get("thermal_image");
  const lapThresh = Number(formData.get("lap_thresh") ?? 80);
  const rawConf = formData.get("conf_thresh");
  const confThresh = rawConf === null || rawConf === "" ? 0 : Number(rawConf);

  if (!(rgbFile instanceof File) && !(thermalFile instanceof File)) {
    return NextResponse.json(
      { ok: false, error: "Upload an RGB image, a thermal image, or both." },
      { status: 400 }
    );
  }

  const options = process.env.HF_TOKEN
    ? { token: process.env.HF_TOKEN as `hf_${string}` }
    : undefined;

  try {
    const client = await Client.connect(normalizeSpaceTarget(spaceUrl), options);
    const response = await client.predict("/gradio_analyze", [
      rgbFile instanceof File ? handle_file(rgbFile) : null,
      thermalFile instanceof File ? handle_file(thermalFile) : null,
      confThresh,
      lapThresh
    ]);

    const payload = firstResult(response.data);
    return NextResponse.json(payload);
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
