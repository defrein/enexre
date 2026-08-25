CREATE CONSTRAINT chemical_mesh_id IF NOT EXISTS
FOR (c:Chemical) REQUIRE c.mesh_id IS UNIQUE;

CREATE CONSTRAINT disease_mesh_id IF NOT EXISTS
FOR (d:Disease) REQUIRE d.mesh_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///chemical_nodes.csv' AS row
MERGE (c:Chemical {mesh_id: row.mesh_id})
SET c.label = row.label,
    c.mention_examples = row.mention_examples;

LOAD CSV WITH HEADERS FROM 'file:///disease_nodes.csv' AS row
MERGE (d:Disease {mesh_id: row.mesh_id})
SET d.label = row.label,
    d.mention_examples = row.mention_examples;

LOAD CSV WITH HEADERS FROM 'file:///cid_edges.csv' AS row
MATCH (c:Chemical {mesh_id: row.chemical_id})
MATCH (d:Disease {mesh_id: row.disease_id})
MERGE (c)-[r:CID {pmid: row.pmid, chemical_id: row.chemical_id, disease_id: row.disease_id}]->(d)
SET r.confidence = toFloat(row.confidence),
    r.source = row.source,
    r.chemical_mentions = row.chemical_mentions,
    r.disease_mentions = row.disease_mentions,
    r.evidence = row.evidence;
