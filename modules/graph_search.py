import heapq
from collections import deque

# Graph Definition (Adjacency List with Weights/Costs)
# Edges ka cost matlab us skill ko seekhne mein kitna time/effort lagega
CAREER_GRAPH = {
    'Python': {'Pandas': 1, 'SQL': 2},
    'Pandas': {'Statistics': 2, 'Machine Learning': 4},
    'SQL': {'Visualization': 1, 'Machine Learning': 3},
    'Statistics': {'Machine Learning': 2},
    'Visualization': {'Data Scientist': 3, 'Machine Learning': 2},
    'Machine Learning': {'Data Scientist': 1},
    'Data Scientist': {}
}

# Heuristic Function for A* (Estimated distance to Target: Data Scientist)
# Ye humne manually estimate kiya hai (jaise Target se kitna door hain)
HEURISTIC = {
    'Python': 4,
    'Pandas': 3,
    'SQL': 3,
    'Statistics': 2,
    'Visualization': 2,
    'Machine Learning': 1,
    'Data Scientist': 0
}
def heuristic(node, goal):
    """Admissible heuristic for any target node."""
    return 0


# 1. BFS (Breadth-First Search) - Shortest path by number of edges
def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = set([start])
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        
        if node == goal:
            return path, len(path) - 1 # Path and number of edges
        
        for neighbor in graph.get(node, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return None, 0

# 2. DFS (Depth-First Search) - Explores deeply
def dfs(graph, start, goal, path=None, visited=None):
    if path is None: path = [start]
    if visited is None: visited = set()
    
    visited.add(start)
    
    if start == goal:
        return path, len(path) - 1
        
    for neighbor in graph.get(start, {}):
        if neighbor not in visited:
            new_path, cost = dfs(graph, neighbor, goal, path + [neighbor], visited)
            if new_path:
                return new_path, cost
    return None, 0

# 3. A* (A-Star) - Informed Search using cost + heuristic
def a_star(graph, start, goal):
    # Priority Queue: (f_score, g_score, current_node, path)
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()
    
    while open_set:
        # Sabse choti f_score wala node nikalo
        f_score, g_score, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path, g_score
            
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in visited:
                new_g_score = g_score + weight
                new_f_score = new_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, path + [neighbor]))
                
    return None, 0

def calculate_path_cost(graph, path):
    """Calculate total weighted cost of a career path."""
    if not path:
        return None

    total_cost = 0

    for current, next_node in zip(path, path[1:]):
        total_cost += graph[current][next_node]

    return total_cost
