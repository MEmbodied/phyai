export const SetupCard = ({ rows = {} }) => {
    const entries = Object.entries(rows)
    const [selections, setSelections] = React.useState(() =>
        Object.fromEntries(entries.map(([k]) => [k, 0]))
    )

    const isTuple = (opt) => Array.isArray(opt)
    const labelOf = (opt) => (isTuple(opt) ? opt[0] : opt)

    let command = ""
    for (const [key, options] of entries) {
        if (options.length > 0 && isTuple(options[0])) {
            const idx = selections[key] ?? 0
            command = options[idx]?.[1] || ""
            break
        }
    }

    const setSelection = (key, i) => {
        setSelections((prev) => ({ ...prev, [key]: i }))
    }

    return (
        <div className="not-prose rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden bg-white dark:bg-zinc-900">
            {entries.map(([key, options]) => {
                const idx = selections[key] ?? 0
                const interactive = options.length > 1
                return (
                    <div key={key} className="flex border-b border-zinc-200 dark:border-zinc-800">
                        <div className="w-44 shrink-0 flex items-center px-4 py-3 text-sm text-zinc-600 dark:text-zinc-400">
                            {key}
                        </div>
                        <div className="flex flex-1 gap-2 p-2">
                            {options.map((opt, i) => {
                                const lbl = labelOf(opt)
                                const selected = idx === i
                                return (
                                    <button
                                        key={lbl}
                                        onClick={() => interactive && setSelection(key, i)}
                                        className={`flex-1 rounded-md px-3 py-2 text-sm transition-colors ${
                                            selected
                                                ? "bg-[#003399] dark:bg-[#2563EB] text-white font-medium"
                                                : "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-200 hover:bg-zinc-200 dark:hover:bg-zinc-700"
                                        } ${interactive ? "cursor-pointer" : "cursor-default"}`}
                                    >
                                        {lbl}
                                    </button>
                                )
                            })}
                        </div>
                    </div>
                )
            })}
            <div className="flex">
                <div className="w-44 shrink-0 flex items-center px-4 py-3 text-sm text-zinc-600 dark:text-zinc-400">
                    Run this Command:
                </div>
                <div className="flex-1 p-2">
                    <pre className="m-0 px-3 py-2 rounded-md bg-zinc-100 dark:bg-zinc-800 text-sm whitespace-pre-wrap text-zinc-800 dark:text-zinc-200">
                        {command || "—"}
                    </pre>
                </div>
            </div>
        </div>
    )
}
