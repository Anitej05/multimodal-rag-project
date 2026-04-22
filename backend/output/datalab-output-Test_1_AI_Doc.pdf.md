7

Chapter Artificial Intelligence

Search starting _ intermediate _ SEARCH Informed state state. Its in AI is the process of Search to a goal state by commonly three transitioning navigating type from such as: through 8 search, Greedy using estimated distance to the Best-first search: Focuses heuristic, expanding the state with the clozest goal. search: A Bimpler vergion of best-first only the heuristic without considering solely on the

Jou can about about search solving compass Informed exploring is to the towards which problems be in the form of: goal. paths search in AI is a efficiently when navigating a complex the solution much algorithm that uses additional Heuristics: Estimates of how close & given state every path  Informed search is a thegproblem space t _aketioaltefdeciaioog Its like powerful to explore first Thiftenfaeoistion faster having = than a technique puzzle; guiding map type blindly and for of past costs. algorithms an accurate heuristic. and complex problems. more likely to lead to the Benefits of informed search: Goal-directedness: Focuses on Accuracy: Can find optimal solutions when using nodes explored compared to uninformed search Speed: Finds solutions faster; especially in larze Efficiency: Significantly reduces the number of goal. paths that are

Cost functions: Values assigned to actions o Applications:

states, indicating their desirability Informed search algorithms are used in a wide

about the problem domain. Domain knowledge: Specific rules or insights range of AI applications, including: Path planning: Robots navigating mazes, self-

Bgorithnes harnessing this information, informed search can significantly reduce the search space driving cars finding routes, etc Game playing: AI players evaluating moves In

and complex problems find solutions faster; especially in large and chess, Go, etc Scheduling and resource allocation: Optimizing

The core Working: comparing nodes in the search space based on priority function. This function typically combines: principle of informed search involves a Examples: solutions in complex task scheduling, resource allocation, etc: Planning and decision-making: Finding optimal decision-making scenarios:

has been Cost of the path Heuristic estimate of remaining estimate of how much effort it will take to reach expended to reach the current state. s0 far: Measures how much effort cost: An the that blindly about the problem space to explore concrete examples ofhow informed search plays & role Informed search algorithms in Al utilize knowledge optimal solution, unlike uninformed algorithms guide every possibility Here are some their seareh for

first, the include: directions: Examples = goal leading of from the current state: 'popular informed The node with the lowest priority score is explored the search towards the most promising search algorithms in various A[ analyze the applications: junction. This guides the robot towards the maze. Informed seareh algorithms like A* 1. Path finding: Imagine a robot navigating a remaining distance to the exit point at each maze layout and estimate the goal can

A search:* Combines the optimal path with the minimum cost and heuristic to find overall cost. much faster than randomly exploring every path



7.2 or mini-max gearch and 2, Game Playing: Al players in smarter and more strategic gameplay algorithms evaluate potential moves based o their predicted utility in the future, leading to Go employ informed search techniques like games alpha-beta pruning These like chess then possible algorithms, Comptbten Gike paths find a golution if one exists. generating its successors This complete, meaning that they are guaranteed %e are eventually considered  breadth-first Some  uninformed Artiticlal + enburea ' search, Intelligence that = search ate al}

patterns, road closures, and distance to 3.Route Planning: Your favorite navigation apps wouldn  be so efficient without informed search: your route while considering real-time updates Algorithms like A* consider factors like traffic optimize goal state. problems. This is because can be inefficient, especially for Inefficiency: Uninformed search irrelevant or dead-end paths before they large may explore or finding " algorithma complex many the

4. with multiple tasks and dependencies? Informed Scheduling and Planning: Planning a schedule algorithms: Here are some common uninformed search

production lines. actions, minimizing potential conflicts and search can help choose the best sequence of from project management to scheduling factory maximizing efficiency This applies to everything to the goal slow for large problems states at the current level before Breadth-first search (BFS): BFS explores next level: This guarantees that the shortest state will be found; but it can moving on to be all tbe path very the

5. Machine Learning: Believe it Or not, even

some machine learning algorithms utilize

informed search reinforcement learning algorithms employ A techniques. For example, some * 2 5

search to explore the state space during training,

policies: leading to faster convergence and more optimal 3 7 4 56

These are just a few examples, and the applications DFS: 1,3,7, 6,5,2,4 BFS: 1,2,5,3,4,6,7

Remember; the to uninformed search methods. key to make more efficient and optimal decisions utilizes additional knowledge about the problem of informed search extend far beyond these scenarios. point is that informed search compared space as deep problems, but it can also Depth-first search (DFS): DFS explores one_ Breadth first search (BFS) algorithm another path. This can be faster than BFS for some as possible before backtracking and trying get stuck in dead-end path

(ii) Unformed Search paths.

a and the problem space without inforobaeion pbout theh problen beyoidc your way around until you find the exit. technique in artificial intelligence (AI) for state, Its like exploring a maze blindfolded, Uninformed search; also known as blind search, is possible actions that can be taken from that beyond the initial state knowledge exploring feeling or a actions varies. the cost of each path and always expands the Uniform-cost search (UCS): UCS keeps trackof than BFS or DFS for problems where the cost of with the lowest cost, This can be more eflicient path

Characteristics of uninformed search: 6

the problem No domain cost of different actions, transitiong between them. algorithms only rely o the basic additional information about the goal state  Slyotehatie exploration: Uninformed search way, typically by expanding one state ata time and Wgorithis Gxplore the search space in a systeenacic knowledge: Uninformed They space, such a8 the states and the dont have structure of search or the any 3 5 Uniform cost search (UCS) algorithm 3 3

