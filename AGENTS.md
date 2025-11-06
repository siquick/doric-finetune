## Code understanding

Whenever you need to retrieve or understand code context, start with the Nuanced MCP Server. Use the returned metadata to map the call graph before diving into the source.

- Treat each node as a function. `callers` are upstream locations (who invokes it) and `callees` are downstream dependencies (who it invokes). Use this to sketch the subgraph around your target.
- Follow call chains while they add insight or the next hop no longer informs your task. Skip <native> callees; they’re leaf nodes.
- After reviewing the metadata, read the function source using the provided start/end ranges to confirm behavior, understand data flow, and capture nuances the graph can’t show.
- Combine structural findings from the graph with behavioral details from the source in your response so the user gets both relationship context and concrete implementation notes.
