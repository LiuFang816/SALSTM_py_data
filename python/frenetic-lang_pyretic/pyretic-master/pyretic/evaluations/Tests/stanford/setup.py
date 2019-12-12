from Topos import *


def setup_topo(*params):
    return StanfordTopology.StanfordTopo()

def setup_workload(net, n, m):
    hosts = net.topo.hosts()
    hosts = [net.getNodeByName(h) for h in hosts]
    k = min(n,m)
    return (hosts[:n][:k], hosts[n:n+m][:k], ['1M'] * k)

