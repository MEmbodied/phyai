export const ModelCard = ({ title, subtitle, icon, rows = {} }) => {
    const entries = Object.entries(rows)

    const renderValue = (value) => {
        if (value === null || value === undefined) {
            return <span className="text-sm text-zinc-400 dark:text-zinc-600">—</span>
        }
        if (Array.isArray(value)) {
            return (
                <div className="flex flex-wrap gap-1.5">
                    {value.map((v, i) => (
                        <span
                            key={i}
                            className="inline-flex items-center px-2 py-0.5 rounded-md text-[11.5px] font-medium bg-[#003399]/[0.06] text-[#003399] ring-1 ring-inset ring-[#003399]/15 dark:bg-[#60A5FA]/[0.10] dark:text-[#60A5FA] dark:ring-[#60A5FA]/20"
                        >
                            {v}
                        </span>
                    ))}
                </div>
            )
        }
        if (typeof value === "string" || typeof value === "number") {
            return (
                <span className="text-sm text-zinc-800 dark:text-zinc-100 break-words">
                    {value}
                </span>
            )
        }
        return value
    }

    const hasHeader = title || subtitle || icon

    return (
        <div className="not-prose my-6 overflow-hidden rounded-xl bg-white dark:bg-zinc-900 ring-1 ring-zinc-200 dark:ring-zinc-800 shadow-[0_1px_2px_rgb(15_23_42_/_0.04),0_4px_16px_-4px_rgb(15_23_42_/_0.06)] dark:shadow-[0_1px_0_rgb(255_255_255_/_0.04)_inset,0_8px_24px_-8px_rgb(0_0_0_/_0.5)]">
            {hasHeader && (
                <div className="flex items-center gap-3.5 px-5 py-4 bg-zinc-50/60 dark:bg-zinc-800/20 border-b border-zinc-200/80 dark:border-zinc-800/80">
                    {icon && (
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[10px] bg-gradient-to-br from-[#003399] to-[#2563EB] text-white text-lg font-semibold ring-1 ring-inset ring-white/10 shadow-[0_1px_2px_rgb(0_51_153_/_0.25),0_3px_6px_-2px_rgb(0_51_153_/_0.18)]">
                            {icon}
                        </div>
                    )}
                    <div className="min-w-0">
                        {title && (
                            <div className="text-[15px] font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
                                {title}
                            </div>
                        )}
                        {subtitle && (
                            <div className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">
                                {subtitle}
                            </div>
                        )}
                    </div>
                </div>
            )}

            <div>
                {entries.map(([key, value], i) => (
                    <div
                        key={key}
                        className={`flex items-stretch ${
                            i < entries.length - 1
                                ? "border-b border-zinc-100 dark:border-zinc-800/60"
                                : ""
                        }`}
                    >
                        <div className="w-44 shrink-0 flex items-center px-5 py-3 text-[13px] font-medium text-zinc-500 dark:text-zinc-400">
                            {key}
                        </div>
                        <div className="flex-1 flex items-center px-5 py-3 min-w-0">
                            {renderValue(value)}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
