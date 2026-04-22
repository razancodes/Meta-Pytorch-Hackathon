"use client";

import { RAMMonitor } from "./ram-monitor";
import { DiskStorage } from "./disk-storage";
import { AsyncProcesses } from "./async-processes";
import { KernelDirectives } from "./kernel-directives";
import { InterruptButton } from "./interrupt-button";

export function OSConsole() {
  return (
    <div className="flex flex-col gap-3 h-full overflow-y-auto pr-1">
      <div className="grid grid-cols-2 gap-3">
        <RAMMonitor />
        <DiskStorage />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <AsyncProcesses />
        <KernelDirectives />
      </div>
      <InterruptButton />
    </div>
  );
}
