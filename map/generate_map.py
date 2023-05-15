import geojson, json
from typing import Dict, List, Tuple, Union, Any, Set
from haversine import haversine
from sys import float_info
from queue import Queue

class Coordinate:
    lat: float
    long: float

    def __init__(self, lat: float, long: float):
        self.lat = lat
        self.long = long
    
    def to_json(self) -> Dict[str, float]:
        return {
            "lat": self.lat,
            "long": self.long
        }
    
    def __get_lat_dis_meters(self, lat1: float, lat2: float) -> float:
        return haversine((lat1, 0), (lat2, 0)) * 1000

    def __get_long_dis_meters(self, long1: float, long2: float) -> float:
        return haversine((0, long1), (0, long2)) * 1000

    def to_rel_coord(self, refCoord) -> None:
        assert(isinstance(refCoord, Coordinate))
        self.lat = self.__get_lat_dis_meters(refCoord.lat, self.lat)
        self.long = self.__get_long_dis_meters(refCoord.long, self.long)

    def __eq__(self, other):
        return isinstance(other, Coordinate) and self.lat == other.lat and self.long == other.long
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(f"{self.lat}-{self.long}")

class Road:
    start: Coordinate
    end: Coordinate
    connIndexStart: int
    connCountStart: int
    connIndexEnd: int
    connCountEnd: int
    index: int

    def __init__(self, start: Coordinate, end: Coordinate):
        self.start = start
        self.end = end
        self.connIndexStart = -1
        self.connCountStart = 0
        self.connIndexEnd = -1
        self.connCountEnd = 0
        self.index = -1

    def to_json(self) -> Dict[str, Any]:
        return {
            "start": self.start.to_json(),
            "end": self.end.to_json(),
            "connIndexStart": self.connIndexStart,
            "connCountStart": self.connCountStart,
            "connIndexEnd": self.connIndexEnd,
            "connCountEnd": self.connCountEnd
        }
    
    def __eq__(self, other):
        return isinstance(other, Road) and self.start == other.start and self.end == other.end
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.start) ^ hash(self.end)

class Map:
    roads: List[Road]
    connectionsMap: Dict[Coordinate, Set[int]]
    connectionRoadIndexList: List[int]
    
    minLat: float
    maxLat: float
    minLong: float
    maxLong: float

    def __init__(self):
        self.roads = list()
        self.connectionsMap = dict()
        self.connectionRoadIndexList = list()
        
        self.minLat = float_info.max
        self.maxLat = 0
        self.minLong = float_info.max
        self.maxLong = 0
    
    def find_min_max_lat_long(self):
        for road in self.roads:
            self.minLat = min(self.minLat, road.start.lat)
            self.minLat = min(self.minLat, road.end.lat)
            self.maxLat = max(self.maxLat, road.start.lat)
            self.maxLat = max(self.maxLat, road.end.lat)
            self.minLong = min(self.minLong, road.start.long)
            self.minLong = min(self.minLong, road.end.long)
            self.maxLong = max(self.maxLong, road.start.long)
            self.maxLong = max(self.maxLong, road.end.long)

    def to_json(self) -> Dict[str, Any]:
        return {
            "minLat": self.minLat,
            "maxLat": self.maxLat,
            "minLong": self.minLong,
            "maxLong": self.maxLong,
            "roads": [r.to_json() for r in self.roads],
            "connections": self.connectionRoadIndexList
        }

def build_map(features: Dict[str, Any]) -> Map:
    map: Map = Map()
    allroads: Set[Road]
    allroads = set()
    feature: Dict[str, Any]
    for feature in features:
        geometry: Dict[str, Any] = feature["geometry"]
        if geometry["type"] != "LineString":
            continue
        roadPointList: List[Tuple[float, float]] = geometry["coordinates"]
        if(len(roadPointList) < 2):
            print(" Skipping road with only one point.")
            continue
        i: int
        for i in range(1, len(roadPointList)):
            road: Road = Road(Coordinate(roadPointList[i-1][0], roadPointList[i-1][1]), Coordinate(roadPointList[i][0], roadPointList[i][1]))
            if road.start == road.end:
                print(" Skipping road with start == end")
                continue
            
            assert(not (road in allroads))
            allroads.add(road)

            road.index = len(map.roads)
            map.roads.append(road)
    return map

def build_road_connections(map: Map) -> None:
    assert(len(map.connectionsMap) == 0)
    # insert all roads with their coordinates into the map
    for road in map.roads:
        if road.start not in map.connectionsMap:
            map.connectionsMap[road.start] = set()
        if road.end not in map.connectionsMap:
            map.connectionsMap[road.end] = set()
        
        map.connectionsMap[road.start].add(road.index)
        map.connectionsMap[road.end].add(road.index)

def walk_graph(map: Map, subgraph: Set[int], firstRoadIndex: int) -> None:
    # queue holding all to be expanded road indices for the currently expanding subgraph
    roadQueue = Queue()
    # add to q and start walking graph, adding other roads to this graph
    subgraph.add(firstRoadIndex)
    roadQueue.put(firstRoadIndex)
    
    # walk graph (BFS)
    while not roadQueue.empty():
        # 1. take road index from q
        roadIndex = roadQueue.get()
        road = map.roads[roadIndex]
        # 2. expand all connected roads into queue and current subgraph
        for index in map.connectionsMap[road.start]:
            if index not in subgraph:
                subgraph.add(index)
                roadQueue.put(index)
        for index in map.connectionsMap[road.end]:
            if index not in subgraph:
                subgraph.add(index)
                roadQueue.put(index)

def remove_unconnected_sub_graphs(map: Map) -> Map:
    # find all subgraphs
    #  each list is a subgraph
    #  each holding a set of roadIndices
    subgraphs: List[Set(int)]
    subgraphs = list()

    # iterate all roads by index
    for roadIndex in range(0, len(map.roads)):
        # if not yet in any subgraph, add new subgraph
        isNewRoad=True
        for graph in subgraphs:
            if roadIndex in graph:
                isNewRoad=False
                break
        
        if isNewRoad:
            subgraphs.append(set())
            walk_graph(map, subgraphs[len(subgraphs) - 1], roadIndex)
    
    # choose largest (discard all others)
    largestIndex = 0
    largestLen = 0
    for graphIndex in range(0, len(subgraphs)):
        if len(subgraphs[graphIndex]) > largestLen:
            largestLen = len(subgraphs[graphIndex])
            largestIndex = graphIndex
    
    # construct new map
    newmap: Map = Map()
    for roadIndex in subgraphs[largestIndex]:
        road = map.roads[roadIndex]
        road.index = len(newmap.roads)
        newmap.roads.append(road)

    return newmap
    
def construct_index_list(map: Map) -> None:
    assert(len(map.connectionRoadIndexList) == 0)

    for road in map.roads:
        road.connIndexStart = len(map.connectionRoadIndexList)
        road.connCountStart = len(map.connectionsMap[road.start])

        # we always add the current roads index first
        map.connectionRoadIndexList.append(road.index)

        # iterate set of indices for the start coordinate
        for roadIndex in map.connectionsMap[road.start]:
            # add all indices (except our own)
            if roadIndex != road.index:
                map.connectionRoadIndexList.append(roadIndex)

        road.connIndexEnd = len(map.connectionRoadIndexList)
        road.connCountEnd = len(map.connectionsMap[road.end])

        # we always add the current roads index first
        map.connectionRoadIndexList.append(road.index)

        # iterate set of indices for the end coordinate
        for roadIndex in map.connectionsMap[road.end]:
            # add all indices (except our own)
            if roadIndex != road.index:
                map.connectionRoadIndexList.append(roadIndex)

# Transforms all coordinates to relative ones
#  by transforming them to meters
#  and offsetting them to (0, 0)
# Note: Leaves a safety border of 10 meters all around
#       So no coords is directly on the border
def transform_coordinates_to_relative(map: Map) -> None:
    map.find_min_max_lat_long()

    refCoord = Coordinate(map.minLat, map.minLong)

    for road in map.roads:
        road.start.to_rel_coord(refCoord)
        road.end.to_rel_coord(refCoord)
        # Add safety border (offset all coords by (10,10))
        road.start.lat += 10.0
        road.start.long += 10.0
        road.end.lat += 10.0
        road.end.long += 10.0
    
    map.find_min_max_lat_long()
    # Add safety border (leave 10m border)
    map.minLat = 0.0
    map.maxLat += 10.0
    map.minLong = 0.0
    map.maxLong += 10.0

print("Loading input.geojson file...")
with open("input.geojson") as f:
    map: Dict[str, Union[str, List[Dict[str, Any]]]] = geojson.load(f)

print("Converting to map...")
map: Map = build_map(map["features"])
print(f" Found {len(map.roads)} road pieces.")

print("Building connections...")
build_road_connections(map)
print(f" Found {len(map.connectionsMap)} coordinates.")

print("Detecting unconnected subgraphs...")
map = remove_unconnected_sub_graphs(map)
print(f" Keeping largest connected subgraph with {len(map.roads)} road pieces.")

print("Re-Building connections...")
build_road_connections(map)
print(f" Found {len(map.connectionsMap)} coordinates.")

print("Constructing index list...")
construct_index_list(map)
print(f" Found {len(map.connectionRoadIndexList)} connections.")

print("Converting coordinates...")
transform_coordinates_to_relative(map)
print(f" Suggested map size: {map.maxLat}x{map.maxLong} meters.")

print("Exporting results to output.json...")
with open('output.json', 'w') as outfile:
    outfile.write(json.dumps(map.to_json()))
print("Done.")