## Requirements

```
pip install --user geojson haversine
```

1. Go to https://overpass-turbo.eu/
2. Choose the region of interest on the map
3. See query.txt for the used query
4. Export as geojson
5. Save into the map directory as input.geojson
6. Run python generate_map.py from the map directory
7. => output.json