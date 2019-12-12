"""Compute the strong components of a digraph in O(E + V)"""

 #  Execution:    java KosarajuSharirSCC filename.txt
 #  Dependencies: Digraph.java TransitiveClosure.java StdOut.java In.java
 #  Data files:   http://algs4.cs.princeton.edu/42digraph/tinyDG.txt
 #
 #  Compute the strongly-connected components of a digraph using the
 #  Kosaraju-Sharir algorithm.
 #
 #  Runs in O(E + V) time.
 #
 #  % java KosarajuSCC tinyDG.txt
 #  5 components
 #  1 
 #  0 2 3 4 5 
 #  9 10 11 12 
 #  6 8 
 #  7
 #
 #  % java KosarajuSharirSCC mediumDG.txt 
 #  10 components
 #  21 
 #  2 5 6 8 9 11 12 13 15 16 18 19 22 23 25 26 28 29 30 31 32 33 34 35 37 38 39 40 42 43 44 46 47 48 49 
 #  14 
 #  3 4 17 20 24 27 36 
 #  41 
 #  7 
 #  45 
 #  1 
 #  0 
 #  10 
 #
 #  % java -Xss50m KosarajuSharirSCC mediumDG.txt 
 #  25 components
 #  7 11 32 36 61 84 95 116 121 128 230   ...
 #  28 73 80 104 115 143 149 164 184 185  ...
 #  38 40 200 201 207 218 286 387 418 422 ...
 #  12 14 56 78 87 103 216 269 271 272    ...
 #  42 48 112 135 160 217 243 246 273 346 ...
 #  46 76 96 97 224 237 297 303 308 309   ...
 #  9 15 21 22 27 90 167 214 220 225 227  ...
 #  74 99 133 146 161 166 202 205 245 262 ...
 #  43 83 94 120 125 183 195 206 244 254  ...
 #  1 13 54 91 92 93 106 140 156 194 208  ...
 #  10 39 67 69 131 144 145 154 168 258   ...
 #  6 52 66 113 118 122 139 147 212 213   ...
 #  8 127 150 182 203 204 249 367 400 432 ...
 #  63 65 101 107 108 136 169 170 171 173 ...
 #  55 71 102 155 159 198 228 252 325 419 ...
 #  4 25 34 58 70 152 172 196 199 210 226 ...
 #  2 44 50 88 109 138 141 178 197 211    ...
 #  57 89 129 162 174 179 188 209 238 276 ...
 #  33 41 49 119 126 132 148 181 215 221  ...
 #  3 18 23 26 35 64 105 124 157 186 251  ...
 #  5 16 17 20 31 47 81 98 158 180 187    ...
 #  24 29 51 59 75 82 100 114 117 134 151 ...
 #  30 45 53 60 72 85 111 130 137 142 163 ...
 #  19 37 62 77 79 110 153 352 353 361    ...
 #  0 68 86 123 165 176 193 239 289 336   ...
 #
 #*****************************************************************************/

package edu.princeton.cs.algs4

#*
 #  The <tt>KosarajuSharirSCC</tt> class represents a data type for 
 #  determining the strong components in a digraph.
 #  The <em>id</em> operation determines in which strong component
 #  a given vertex lies; the <em>areStronglyConnected</em> operation
 #  determines whether two vertices are in the same strong component;
 #  and the <em>count</em> operation determines the number of strong
 #  components.

 #  The <em>component identifier</em> of a component is one of the
 #  vertices in the strong component: two vertices have the same component
 #  identifier if and only if they are in the same strong component.

 #  <p>
 #  This implementation uses the Kosaraju-Sharir algorithm.
 #  The constructor takes time proportional to <em>V</em> + <em>E</em>
 #  (in the worst case),
 #  where <em>V</em> is the number of vertices and <em>E</em> is the number of edges.
 #  Afterwards, the <em>id</em>, <em>count</em>, and <em>areStronglyConnected</em>
 #  operations take constant time.
 #  For alternate implementations of the same API, see
 #  {@link TarjanSCC} and {@link GabowSCC}.
 #  <p>
 #  For additional documentation,
 #  see <a href="http://algs4.cs.princeton.edu/42digraph">Section 4.2</a> of
 #  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 #
 #  @author Robert Sedgewick
 #  @author Kevin Wayne
 #/
public class KosarajuSharirSCC:
  private boolean[] marked;     # marked[v] = has vertex v been visited?
  private int[] id;             # id[v] = id of strong component containing v
  private count;            # number of strongly-connected components

    #*
     # Computes the strong components of the digraph <tt>G</tt>.
     # @param G the digraph
     #/
  public KosarajuSharirSCC(Digraph G):

      # compute reverse postorder of reverse graph
      DepthFirstOrder dfs = new DepthFirstOrder(G.reverse())

      # run DFS on G, using reverse postorder to guide calculation
      marked = new boolean[G.V()]
      id = new int[G.V()]
      for (int v : dfs.reversePost()):
          if !marked[v]):
              dfs(G, v)
              count += 1

      # check that id[] gives strong components
      assert check(G)

  # DFS on graph G
  def _dfs(Digraph G, v): 
      marked[v] = True
      id[v] = count
      for (int w : G.adj(v)):
          if !marked[w]) dfs(G, w)

    #*
     # Returns the number of strong components.
     # @return the number of strong components
     #/
  def count():
      return count

    #*
     # Are vertices <tt>v</tt> and <tt>w</tt> in the same strong component?
     # @param v one vertex
     # @param w the other vertex
     # @return <tt>true</tt> if vertices <tt>v</tt> and <tt>w</tt> are in the same
     #     strong component, and <tt>false</tt> otherwise
     #/
  def stronglyConnected(int v, w):
      return id[v] == id[w]

    #*
     # Returns the component id of the strong component containing vertex <tt>v</tt>.
     # @param v the vertex
     # @return the component id of the strong component containing vertex <tt>v</tt>
     #/
  def id(int v):
      return id[v]

  # does the id[] array contain the strongly connected components?
  def _check(Digraph G):
      TransitiveClosure tc = new TransitiveClosure(G)
      for (int v = 0; v < G.V(); v += 1):
          for (int w = 0; w < G.V(); w += 1):
              if stronglyConnected(v, w) != (tc.reachable(v, w) and tc.reachable(w, v)))
                  return False
      return True

    #*
     # Unit tests the <tt>KosarajuSharirSCC</tt> data type.
     #/
  def main(String[] args):
      In in = new In(args[0])
      Digraph G = new Digraph(in)
      KosarajuSharirSCC scc = new KosarajuSharirSCC(G)

      # number of connected components
      M = scc.count()
      prt.write(M + " components")

      # compute list of vertices in each strong component
      Queue<Integer>[] components = (Queue<Integer>[]) new Queue[M]
      for (int i = 0; i < M; i += 1):
          components[i] = new Queue<Integer>()
      for (int v = 0; v < G.V(); v += 1):
          components[scc.id(v)].enqueue(v)

      # print results
      for (int i = 0; i < M; i += 1):
          for (int v : components[i]):
              StdOut.print(v + " ")
          prt.write()

# EDIT THIS
"""Compute connected components using depth first search."""

class KosarajuSharirSCC(object):
  """Computes the strong components of a directed graph G. Runs in O(E + V) time."""

  def __init__(self, G):
    self._marked = [False]*G.V() # marked[v] = has vertex v been marked?
    self._id = [0]*G.V()   # id[v] = id of connected component containing v
    self._size = [0]*G.V() # size[id] = number of vertices in given component
    self._count = 0        # number of connected components
    dfs = DepthFirstGraphOrder(G.reverse())
    for v in range(dfs.reversePost():
    for v in range(G.V()): # TBD remove this
      if not self._marked[v]:
        self._dfs(G, v)
        self._count += 1

  def _dfs(self, G, v):
    """depth-first search"""
    self._marked[v] = True
    self._id[v] = self._count
    self._size[self._count] += 1
    for w in G.adj(v):
      if not self._marked[w]:
        self._dfs(G, w)

  # Returns the component id of the connected component containing vertex <tt>v</tt>.
  def id(self, v): return self._id[v] 

  # Returns the number of vertices in the connected component containing vertex <tt>v</tt>.
  def size(self, v): return self._size[id[v]]

  # Returns the number of connected components in the graph <tt>G</tt>.
  def count(self): return self._count

  # Returns true if vertices <tt>v</tt> and <tt>w</tt> are in the same
  def connected(self, v, w): return self._id(v) == self._id(w)

  # Returns true if vertices v and w are in the same connected component.
  def areConnected(self, v, w): return self._id(v) == self._id(w)


# Strong Components

# 1980s: easy two-pass linear-time algorithm (Kosaraju-Sharir)
#   * Forgot notes for lecture; developed algorithm in order to teach it!
#   * Later found in Russion scientific lieterature (1972)

# QUESTION: The Kosaraju-Sharir algorithm runs DFS in G^R followed by DFS in G.
# Which of the following modifications will not find the strong components?
# --> Run BFS in G^R followed by DFS in G
#     Run DFS in G^R followed by BFS in G
#     Run DFS in G   followed by BFS in G^R
#     Run DFS in G   followed by DFS in G^R
# EXPLANATION: The DFS in the 1st phase (to compute the reverse postorder) is crucial;
# in the 2nd phase, any algorithm that marks the set of vertices reachable from
# a given vertex will do.

# Copyright 2002-2016, Robert Sedgewick and Kevin Wayne.
# Copyright 2015-2016, DV Klopfenstein, Python implementation
