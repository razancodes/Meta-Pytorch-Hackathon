/**
 * Memex OS-Agent — Preference Pair API for DPO Continuous Learning.
 *
 * POST /api/preferences — Submit a human correction (creates a preference pair).
 * GET  /api/preferences — Retrieve unconsumed preference pairs for DPO training.
 * PATCH /api/preferences — Mark pairs as consumed by a training run.
 */

import { NextRequest, NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";

// Singleton Prisma client (prevents hot-reload connection exhaustion)
const globalForPrisma = globalThis as unknown as { prisma: PrismaClient };
const prisma = globalForPrisma.prisma ?? new PrismaClient();
if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;

// ═══════════════════════════════════════════════════════════════════
// POST — Submit a correction
// ═══════════════════════════════════════════════════════════════════

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();

        // Validate required fields
        const { originalPrompt, rejectedResponse, chosenResponse } = body;
        if (!originalPrompt || !rejectedResponse || !chosenResponse) {
            return NextResponse.json(
                {
                    error: "Missing required fields",
                    required: ["originalPrompt", "rejectedResponse", "chosenResponse"],
                },
                { status: 400 }
            );
        }

        // Validate JSON structure of responses
        try {
            JSON.parse(rejectedResponse);
            JSON.parse(chosenResponse);
        } catch {
            return NextResponse.json(
                { error: "rejectedResponse and chosenResponse must be valid JSON strings" },
                { status: 400 }
            );
        }

        const pair = await prisma.preferencePair.create({
            data: {
                originalPrompt,
                rejectedResponse,
                chosenResponse,
                difficulty: body.difficulty ?? "medium",
                typology: body.typology ?? "structuring",
                correctionReason: body.correctionReason ?? null,
            },
        });

        return NextResponse.json(
            { id: pair.id, message: "Preference pair saved" },
            { status: 201 }
        );
    } catch (error) {
        console.error("[POST /api/preferences] Error:", error);
        return NextResponse.json(
            { error: "Internal server error" },
            { status: 500 }
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// GET — Retrieve unconsumed pairs for training
// ═══════════════════════════════════════════════════════════════════

export async function GET(request: NextRequest) {
    try {
        const { searchParams } = new URL(request.url);
        const limit = Math.min(parseInt(searchParams.get("limit") ?? "100"), 500);
        const includeConsumed = searchParams.get("include_consumed") === "true";

        const pairs = await prisma.preferencePair.findMany({
            where: includeConsumed ? {} : { consumed: false },
            orderBy: { createdAt: "asc" },
            take: limit,
        });

        return NextResponse.json({
            pairs,
            count: pairs.length,
            hasMore: pairs.length === limit,
        });
    } catch (error) {
        console.error("[GET /api/preferences] Error:", error);
        return NextResponse.json(
            { error: "Internal server error" },
            { status: 500 }
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// PATCH — Mark pairs as consumed by a training run
// ═══════════════════════════════════════════════════════════════════

export async function PATCH(request: NextRequest) {
    try {
        const body = await request.json();
        const { pairIds, runId } = body;

        if (!pairIds || !Array.isArray(pairIds) || !runId) {
            return NextResponse.json(
                {
                    error: "Missing required fields",
                    required: ["pairIds (string[])", "runId (string)"],
                },
                { status: 400 }
            );
        }

        const updated = await prisma.preferencePair.updateMany({
            where: { id: { in: pairIds } },
            data: {
                consumed: true,
                consumedByRunId: runId,
                consumedAt: new Date(),
            },
        });

        return NextResponse.json({
            updated: updated.count,
            runId,
        });
    } catch (error) {
        console.error("[PATCH /api/preferences] Error:", error);
        return NextResponse.json(
            { error: "Internal server error" },
            { status: 500 }
        );
    }
}
