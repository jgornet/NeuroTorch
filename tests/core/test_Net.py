import unittest
from neurotorch.nets.NetCollector import NetCollector
import neurotorch.nets


class TestNet(unittest.TestCase):
    def test_load_net(self):
        test = NetCollector().get_module("RSUNet")
