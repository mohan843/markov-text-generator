import unittest

from generator import MarkovTextGenerator, detokenize, tokenize


class MarkovGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.text = "The cat sees the dog. The dog sees the cat. The cat learns quickly."

    def test_tokenize_and_detokenize(self):
        tokens = tokenize("Hello, world!")
        self.assertEqual(tokens, ["Hello", ",", "world", "!"])
        self.assertEqual(detokenize(tokens), "Hello, world!")

    def test_fit_builds_transitions(self):
        model = MarkovTextGenerator(order=2).fit(self.text)
        self.assertIn(("The", "cat"), model.transitions_)
        self.assertGreater(len(model.vocabulary()), 5)

    def test_generation_respects_length(self):
        model = MarkovTextGenerator(order=2, seed=4).fit(self.text)
        generated = model.generate(max_tokens=8, start="The cat")
        self.assertLessEqual(len(tokenize(generated)), 8)
        self.assertTrue(generated.startswith("The cat"))

    def test_seed_makes_generation_reproducible(self):
        first = MarkovTextGenerator(order=2, seed=10).fit(self.text).generate(12)
        second = MarkovTextGenerator(order=2, seed=10).fit(self.text).generate(12)
        self.assertEqual(first, second)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            MarkovTextGenerator(order=0)
        with self.assertRaises(ValueError):
            MarkovTextGenerator(order=3).fit("one two")
        model = MarkovTextGenerator().fit(self.text)
        with self.assertRaises(ValueError):
            model.generate(0)
        with self.assertRaises(ValueError):
            model.generate(10, start="unknown phrase")


if __name__ == "__main__":
    unittest.main()
