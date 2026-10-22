# Neo4j graph backend (Docker)

This spins up a local Neo4j instance so the pipeline can use a real graph database instead
of the in-memory NetworkX fallback.

## Start

```bash
cd docker
docker compose up -d
```

- Browser UI: http://localhost:7474
- Bolt endpoint: `bolt://localhost:7687`
- Credentials: `neo4j` / `graphrag_pw`

Wait until `docker compose ps` shows the container as `healthy` (first boot pulls the
image and initialises the store).

## Point the pipeline at Neo4j

```bash
export GRAPHRAG_GRAPH=neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=graphrag_pw

# install the driver (part of requirements.txt)
pip install neo4j

# (re)load the graph into Neo4j and build the vector index
python -m scripts.generate_data
python -m scripts.build_index

# query — graph traversals now run against Neo4j
python -m scripts.run_query --compare \
  "What are the failure modes for pumps downstream of Separator V-202?"
```

If Neo4j is unreachable the pipeline automatically falls back to NetworkX (set
`GRAPHRAG_GRAPH_FALLBACK=false` to make that a hard error instead).

## Inspect the graph in the browser

Useful Cypher once the data is loaded:

```cypher
// overview
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC;

// what does Separator V-202 feed, 2 hops downstream?
MATCH path = (s:Component)-[:CONNECTED_TO*1..2]->(c:Component)
WHERE s.name CONTAINS 'V-202'
RETURN path;

// SOPs governing failure modes seen on pumps
MATCH (c:Component {type:'Centrifugal_Pump'})<-[:PERFORMED_ON]-(w:WorkOrder)
      -[:TRIGGERED_BY]->(fm:FailureMode)<-[:GOVERNS_REPAIR]-(s:SOP)
RETURN DISTINCT c.comp_id, fm.failure_code, s.sop_id;
```

## Stop

```bash
docker compose down      # keep data volume
docker compose down -v   # wipe data volume
```
