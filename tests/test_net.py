import unittest
from neurotorch.nets.NetCollector import NetCollector


class TestNet(unittest.TestCase):
    def test_load_net(self):
        test = NetCollector().get_module("RSUNet")
