#*****************************************************************************
 #  Compilation:  javac TarjanSCC.java
 #  Execution:    Java TarjanSCC V E
 #  Dependencies: Digraph.java Stack.java TransitiveClosure.java StdOut.java
 #
 #  Compute the strongly-connected components of a digraph using 
 #  Tarjan's algorithm.
 #
 #  Runs in O(E + V) time.
 #
 #  % java TarjanSCC tinyDG.txt
 #  5 components
 #  1 
 #  0 2 3 4 5
 #  9 10 11 12
 #  6 8
 #  7 
 #
 #*****************************************************************************/

package edu.princeton.cs.algs4

#*
 #  The <tt>TarjanSCC</tt> class represents a data type for 
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
 #  This implementation uses Tarjan's algorithm.
 #  The constructor takes time proportional to <em>V</em> + <em>E</em>
 #  (in the worst case),
 #  where <em>V</em> is the number of vertices and <em>E</em> is the number of edges.
 #  Afterwards, the <em>id</em>, <em>count</em>, and <em>areStronglyConnected</em>
 #  operations take constant time.
 #  For alternate implementations of the same API, see
 #  {@link KosarajuSharirSCC} and {@link GabowSCC}.
 #  <p>
 #  For additional documentation,
 #  see <a href="http://algs4.cs.princeton.edu/42digraph">Section 4.2</a> of
 #  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 #
 #  @author Robert Sedgewick
 #  @author Kevin Wayne
 #/
public class TarjanSCC:

  private boolean[] marked;        # marked[v] = has v been visited?
  private int[] id;                # id[v] = id of strong component containing v
  private int[] low;               # low[v] = low number of v
  private pre;                 # preorder number counter
  private count;               # number of strongly-connected components
  private Stack<Integer> stack


    #*
     # Computes the strong components of the digraph <tt>G</tt>.
     # @param G the digraph
     #/
  public TarjanSCC(Digraph G):
      marked = new boolean[G.V()]
      stack = new Stack<Integer>()
      id = new int[G.V()]
      low = new int[G.V()]
      for (int v = 0; v < G.V(); v += 1):
          if !marked[v]) dfs(G, v)

      # check that id[] gives strong components
      assert check(G)

  def _dfs(Digraph G, v): 
      marked[v] = True
      low[v] = pre += 1
      min = low[v]
      stack.push(v)
      for (int w : G.adj(v)):
          if !marked[w]) dfs(G, w)
          if low[w] < min) min = low[w]
      if min < low[v]):
          low[v] = min
          return
      w
      do:
          w = stack.pop()
          id[w] = count
          low[w] = G.V()
      } while (w != v)
      count += 1


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
     # Unit tests the <tt>TarjanSCC</tt> data type.
     #/
  def main(String[] args):
      In in = new In(args[0])
      Digraph G = new Digraph(in)
      TarjanSCC scc = new TarjanSCC(G)

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

# Strong Components

# 1972: linear-time DFS algorithm (Tarjan).
#   Classic algorithm
#   Level of difficulty: Algs4++
#   Demonstrated broad applicability and importance of DFS

#*****************************************************************************
 #  Copyright 2002-2016, Robert Sedgewick and Kevin Wayne.
 #
 #  This file is part of algs4.jar, which accompanies the textbook
 #
 #      Algorithms, 4th edition by Robert Sedgewick and Kevin Wayne,
 #      Addison-Wesley Professional, 2011, ISBN 0-321-57351-X.
 #      http://algs4.cs.princeton.edu
 #
 #
 #  algs4.jar is free software: you can redistribute it and/or modify
 #  it under the terms of the GNU General Public License as published by
 #  the Free Software Foundation, either version 3 of the License, or
 #  (at your option) any later version.
 #
 #  algs4.jar is distributed in the hope that it will be useful,
 #  but WITHOUT ANY WARRANTY; without even the implied warranty of
 #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 #  GNU General Public License for more details.
 #
 #  You should have received a copy of the GNU General Public License
 #  along with algs4.jar.  If not, see http://www.gnu.org/licenses.
 #*****************************************************************************/
