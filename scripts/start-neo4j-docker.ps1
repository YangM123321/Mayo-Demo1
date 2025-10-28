docker pull neo4j:5.22
& { docker rm -f mayo-neo4j } -ErrorAction SilentlyContinue | Out-Null
docker run --name mayo-neo4j -d -p 7474:7474 -p 7687:7687 -e "NEO4J_AUTH=neo4j/testpass" neo4j:5.22
