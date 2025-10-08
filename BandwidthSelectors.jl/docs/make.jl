using Documenter, BandwidthSelectors

makedocs(
    sitename="BandwidthSelectors.jl",
    modules = [BandwidthSelectors],
    format=Documenter.HTML(),
    pages = [
        "Introdution" => "index.md"
    ],
    checkdocs=:none
)

deploydocs(;
    repo="git@github.com/oskarhs/BandwidthSelectors.jl",
    push_preview = true
)