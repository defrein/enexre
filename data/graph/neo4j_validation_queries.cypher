MATCH (n)
WITH labels(n) AS labels, n.mesh_id AS mesh_id, count(n) AS total
WHERE mesh_id IS NOT NULL AND total > 1
RETURN labels, mesh_id, total;

MATCH ()-[r:CID]->()
WHERE r.pmid IS NULL
RETURN count(r) AS relationships_without_pmid;

MATCH ()-[r:CID]->()
WHERE r.confidence IS NULL
RETURN count(r) AS relationships_without_confidence;

MATCH ()-[r:CID]->()
WITH r.chemical_id AS chemical_id, r.disease_id AS disease_id, r.pmid AS pmid, count(r) AS total
WHERE total > 1
RETURN chemical_id, disease_id, pmid, total;

MATCH (a)-[r:CID]->(b)
WHERE NOT a:Chemical OR NOT b:Disease
RETURN count(r) AS invalid_cid_direction;

MATCH ()-[r:CID]->()
RETURN count(r) AS cid_relationships;
