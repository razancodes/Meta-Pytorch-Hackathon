'use client';

import { useEffect, useRef } from 'react';
import styles from './case.module.css';
import type { ReplayStep } from '@/lib/types';

interface Props {
    steps: ReplayStep[];
}

const TOOL_ICONS: Record<string, string> = {
    review_alert: '🔔',
    get_customer_profile: '👤',
    query_transactions: '💳',
    check_watchlist: '🔍',
    trace_network: '🕸',
    check_source_of_funds: '💰',
    write_to_case_file: '💾',
    update_system_prompt: '⚙',
    search_compliance_manual: '📋',
    request_wire_trace: '📡',
    retrieve_async_result: '📥',
    file_sar: '🚨',
    close_alert: '✅',
};

export default function AgentTerminal({ steps }: Props) {
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [steps.length]);

    return (
        <div className={styles.terminalContainer}>
            <div className={styles.terminalBody} ref={scrollRef}>
                {/* Initial prompt */}
                <div className={styles.termEntry}>
                    <div className={styles.termPrompt}>
                        <span className={styles.termPS1}>memex@case:~$</span>
                        <span className={styles.termCmd}>init_investigation</span>
                    </div>
                    <div className={styles.termOutput}>&gt; Environment loaded. Awaiting agent actions.</div>
                </div>

                {/* Step entries */}
                {steps.map((step) => {
                    const icon = TOOL_ICONS[step.action.tool] || '▸';
                    const reward = step.observation.reward ?? 0;
                    const rewardColor = reward >= 0.05 ? '#22C55E' : reward < 0 ? '#D4334A' : '#505055';

                    return (
                        <div
                            key={step.step_number}
                            className={`${styles.termEntry} ${step.observation.done ? styles.termEntryFinal : ''}`}
                        >
                            {/* Step header */}
                            <div className={styles.termStepHeader}>
                                <span className={styles.termStepNum}>STEP {String(step.step_number).padStart(2, '0')}</span>
                                <span className={styles.termReward} style={{ color: rewardColor }}>
                                    {reward >= 0 ? '+' : ''}{reward.toFixed(2)}
                                </span>
                            </div>

                            {/* Reasoning */}
                            {step.reasoning && (
                                <div className={styles.termReasoning}>
                                    <span className={styles.termReasonLabel}>REASONING</span>
                                    <p className={styles.termReasonText}>{step.reasoning}</p>
                                </div>
                            )}

                            {/* Tool call */}
                            <div className={styles.termPrompt}>
                                <span className={styles.termPS1}>memex@case:~$</span>
                                <span className={styles.termCmd}>
                                    {icon} {step.action.tool}
                                </span>
                            </div>

                            {/* Parameters */}
                            {Object.keys(step.action.parameters).length > 0 && (
                                <div className={styles.termParams}>
                                    {Object.entries(step.action.parameters).map(([key, val]) => (
                                        <div key={key} className={styles.termParam}>
                                            <span className={styles.termParamKey}>{key}:</span>
                                            <span className={styles.termParamVal}>
                                                {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Output */}
                            <div className={styles.termOutput}>
                                &gt; {step.observation.message}
                            </div>

                            {/* Terminal marker for final step */}
                            {step.observation.done && (
                                <div className={styles.termFinal}>
                                    <span className={styles.termFinalIcon}>█</span>
                                    EPISODE COMPLETE — FINAL SCORE: {(step.observation.reward ?? 0) >= 0 ? '+' : ''}{(step.observation.reward ?? 0).toFixed(4)}
                                </div>
                            )}
                        </div>
                    );
                })}

                {/* Cursor blink */}
                <div className={styles.termCursor}>
                    <span className={styles.termPS1}>memex@case:~$</span>
                    <span className={styles.termBlink}>█</span>
                </div>
            </div>
        </div>
    );
}
