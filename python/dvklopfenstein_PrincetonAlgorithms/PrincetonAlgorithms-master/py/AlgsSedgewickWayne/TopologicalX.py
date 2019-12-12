#*****************************************************************************
 #  Compilation:  javac TopologicalX.java
 #  Execution:    java TopologicalX V E F
 #  Dependencies: Queue.java Digraph.java
 #
 #  Compute topological ordering of a DAG using queue-based algorithm.
 #  Runs in O(E + V) time.
 #
 #*****************************************************************************/

package edu.princeton.cs.algs4

#*
 #  The <tt>Topological</Xtt> class represents a data type for 
 #  determining a topological order of a directed acyclic graph (DAG).
 #  Recall, a digraph has a topological order if and only if it is a DAG.
 #  The <em>hasOrder</em> operation determines whether the digraph has
 #  a topological order, and if so, the <em>order</em> operation
 #  returns one.
 #  <p>
 #  This implementation uses a nonrecursive, queue-based algorithm.
 #  The constructor takes time proportional to <em>V</em> + <em>E</em>
 #  (in the worst case),
 #  where <em>V</em> is the number of vertices and <em>E</em> is the number of edges.
 #  Afterwards, the <em>hasOrder</em> and <em>rank</em> operations takes constant time;
 #  the <em>order</em> operation takes time proportional to <em>V</em>.
 #  <p>
 #  See {@link DirectedCycle}, {@link DirectedCycleX}, and
 #  {@link EdgeWeightedDirectedCycle} to compute a
 #  directed cycle if the digraph is not a DAG.
 #  See {@link Topological} for a recursive version that uses depth-first search.
 #  <p>
 #  For additional documentation,
 #  see <a href="http://algs4.cs.princeton.edu/42digraph">Section 4.2</a> of
 #  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 #
 #  @author Robert Sedgewick
 #  @author Kevin Wayne
 #/
public class TopologicalX:
  private Queue<Integer> order;     # vertices in topological order
  private int[] rank;               # rank[v] = order where vertex v appers in order

    #*
     # Determines whether the digraph <tt>G</tt> has a topological order and, if so,
     # finds such a topological order.
     # @param G the digraph
     #/
  public TopologicalX(Digraph G):

      # indegrees of remaining vertices
      int[] indegree = new int[G.V()]
      for (int v = 0; v < G.V(); v += 1):
          indegree[v] = G.indegree(v)

      # initialize 
      rank = new int[G.V()]
      order = new Queue<Integer>()
      count = 0

      # initialize queue to contain all vertices with indegree = 0
      Queue<Integer> queue = new Queue<Integer>()
      for (int v = 0; v < G.V(); v += 1)
          if indegree[v] == 0) queue.enqueue(v)

      for (int j = 0; !queue.isEmpty(); j += 1):
          v = queue.dequeue()
          order.enqueue(v)
          rank[v] = count += 1
          for (int w : G.adj(v)):
              indegree[w] -= 1
              if indegree[w] == 0) queue.enqueue(w)

      # there is a directed cycle in subgraph of vertices with indegree >= 1.
      if count != G.V()):
          order = None

      assert check(G)

    #*
     # Returns a topological order if the digraph has a topologial order,
     # and <tt>null</tt> otherwise.
     # @return a topological order of the vertices (as an interable) if the
     #    digraph has a topological order (or equivalently, if the digraph is a DAG),
     #    and <tt>null</tt> otherwise
     #/
  def order():
      return order

    #*
     # Does the digraph have a topological order?
     # @return <tt>true</tt> if the digraph has a topological order (or equivalently,
     #    if the digraph is a DAG), and <tt>false</tt> otherwise
     #/
  def hasOrder():
      return order is not None

    #*
     # The the rank of vertex <tt>v</tt> in the topological order;
     # -1 if the digraph is not a DAG
     # @return the position of vertex <tt>v</tt> in a topological order
     #    of the digraph; -1 if the digraph is not a DAG
     # @throws IndexOutOfBoundsException unless <tt>v</tt> is between 0 and
     #    <em>V</em> &minus; 1
     #/
  def rank(int v):
      validateVertex(v)
      if hasOrder()) return rank[v]
      else:            return -1

  # certify that digraph is acyclic
  def _check(Digraph G):

      # digraph is acyclic
      if hasOrder()):
          # check that ranks are a permutation of 0 to V-1
          boolean[] found = new boolean[G.V()]
          for (int i = 0; i < G.V(); i += 1):
              found[rank(i)] = True
          for (int i = 0; i < G.V(); i += 1):
              if !found[i]):
                  System.err.println("No vertex with rank " + i)
                  return False

          # check that ranks provide a valid topological order
          for (int v = 0; v < G.V(); v += 1):
              for (int w : G.adj(v)):
                  if rank(v) > rank(w)):
                      System.err.printf("%d-%d: rank(%d) = %d, rank(%d) = %d\n",
                                        v, w, v, rank(v), w, rank(w))
                      return False

          # check that order() is consistent with rank()
          r = 0
          for (int v : order()):
              if rank(v) != r):
                  System.err.println("order() and rank() inconsistent")
                  return False
              r += 1


      return True

  # raise an IndexOutOfBoundsException unless 0 <= v < V
  def _validateVertex(int v):
      V = len(rank)
      if v < 0 or v >= V)
          raise new IndexOutOfBoundsException("vertex " + v + " is not between 0 and " + (V-1))

    #*
     # Unit tests the <tt>TopologicalX</tt> data type.
     #/
  def main(String[] args):

      # create random DAG with V vertices and E edges; then add F random edges
      V = Integer.parseInt(args[0])
      E = Integer.parseInt(args[1])
      F = Integer.parseInt(args[2])
      Digraph G = DigraphGenerator.dag(V, E)

      # add F extra edges
      for (int i = 0; i < F; i += 1):
          v = StdRandom.uniform(V)
          w = StdRandom.uniform(V)
          G.addEdge(v, w)

      prt.write(G)

      # find a directed cycle
      TopologicalX topological = new TopologicalX(G)
      if !topological.hasOrder()):
          prt.write("Not a DAG")

      # or give topologial sort
      else:
          StdOut.print("Topological order: ")
          for (int v : topological.order()):
              StdOut.print(v + " ")
          prt.write()


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
