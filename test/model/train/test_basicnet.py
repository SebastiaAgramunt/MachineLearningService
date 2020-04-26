import unittest
from src.model.train.basicnet import BasicNet


class TestBasicNet(unittest.TestCase):
    def setUp(self):
        self.init_obj = BasicNet()

    def test_input_1(self):
        self.assertTrue(self.init_obj.in_size > 0)

    def test_input_2(self):
        self.assertTrue(isinstance(self.init_obj.in_size, int))

    def test_input_3(self):
        self.assertTrue(self.init_obj.out_size > 0)

    def test_input_4(self):
        self.assertTrue(self.init_obj.out_size > 0)


if __name__ == '__main__':
    unittest.main()
