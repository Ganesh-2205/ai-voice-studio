"use server";

import { v2 as cloudinary } from "cloudinary";

import { auth } from "~/lib/auth";
import { headers } from "next/headers";

import { env } from "~/env";
import { db } from "~/server/db";
import { cache } from "react";

cloudinary.config({
  cloud_name: env.CLOUDINARY_CLOUD_NAME,
  api_key: env.CLOUDINARY_API_KEY,
  api_secret: env.CLOUDINARY_API_SECRET,
});

interface UploadVoiceResult {
  success: boolean;
  id?: string;
  s3Key?: string;
  url?: string;
  error?: string;
}

export async function uploadVoice(
  formData: FormData,
): Promise<UploadVoiceResult> {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user?.id) {
      return { success: false, error: "Unauthorized" };
    }

    if (
      !env.CLOUDINARY_CLOUD_NAME ||
      !env.CLOUDINARY_API_KEY ||
      !env.CLOUDINARY_API_SECRET
    ) {
      return { success: false, error: "Cloudinary not configured" };
    }

    const file = formData.get("voice") as File;

    if (!file) {
      return { success: false, error: "No file provided" };
    }

    if (!file.type.startsWith("audio/")) {
      return { success: false, error: "File must be audio" };
    }

    if (file.size > 10 * 1024 * 1024) {
      return { success: false, error: "File must be under 10MB" };
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const uploadResult = await new Promise((resolve, reject) => {
      cloudinary.uploader
        .upload_stream(
          {
            resource_type: "auto",
            folder: `ai-voice-studio/voices/${session.user.id}`,
          },
          (error, result) => {
            if (error) {
              reject(error);
            } else {
              resolve(result);
            }
          },
        )
        .end(buffer);
    }) as any;

    const uploadedVoice = await db.uploadedVoice.create({
      data: {
        name: file.name,
        s3Key: uploadResult.public_id, // We'll keep the field name s3Key for now to avoid schema changes if possible, but store public_id
        url: uploadResult.secure_url,
        userId: session.user.id,
      },
    });

    return {
      success: true,
      id: uploadedVoice.id,
      s3Key: uploadResult.public_id,
      url: uploadResult.secure_url,
    };
  } catch (error) {
    console.error("Voice upload error:", error);
    return { success: false, error: "Failed to upload voice file" };
  }
}

export const getUserUploadedVoices = cache(async () => {
  try {
    const session = await auth.api.getSession({
      headers: await headers(),
    });

    if (!session?.user?.id) {
      return { success: false, error: "Unauthorized", voices: [] };
    }

    const uploadedVoices = await db.uploadedVoice.findMany({
      where: { userId: session.user.id },
      orderBy: { createdAt: "desc" },
    });

    return { success: true, voices: uploadedVoices };
  } catch (error) {
    console.error("Error fetching uploaded voices:", error);
    return {
      success: false,
      error: "Failed to fetch uploaded voices",
      voices: [],
    };
  }
});
