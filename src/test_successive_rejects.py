import unittest
from bandit import SuccessiveRejects

class TestSuccessiveRejects(unittest.TestCase):
    def setUp(self):
        self.bandit = SuccessiveRejects(num_arms=3, T=100)
    
    def test_choose_action(self):
        # Test initial arm selection
        self.assertEqual(self.bandit.choose_action(), 0)
        
        # Test arm selection after some pulls
        self.bandit.num_pulls[0] = 50
        self.assertEqual(self.bandit.choose_action(), 1)
        self.bandit.num_pulls[0] = 100
        self.assertEqual(self.bandit.choose_action(), 1)
        
        # Test arm selection after all arms have been pulled
        self.bandit.num_pulls = [100, 100, 100]
        self.assertEqual(self.bandit.choose_action(), 2)
        
    def test_best_arm(self):
        # Test best arm selection
        
        self.assertEqual(self.bandit.best_arm(), 0)
        
        # Test best arm selection after some arms have been rejected
        self.bandit.cumulative_rewards = [50, 100, 75]
        self.bandit.num_pulls = [100, 100, 100]
        self.bandit.choose_action()
        self.bandit.choose_action()
        self.bandit.choose_action()
        print(self.bandit.active_arms, self.bandit.best_arm())
        self.assertEqual(self.bandit.best_arm(), 1)
        
    def test_update(self):
        # Test reward update
        self.bandit.update(0, 1)
        self.assertEqual(self.bandit.cumulative_rewards[0], 1)
        self.assertEqual(self.bandit.num_pulls[0], 1)
        
    def test_reset(self):
        # Test reset
        self.bandit.cur_round = 2
        self.bandit.cur_arm = 1
        self.bandit.num_pulls = [50, 50, 50]
        self.bandit.active_arms = [0, 1, 2]
        self.bandit.cumulative_rewards = [10, 20, 30]
        
        self.bandit.reset()
        
        self.assertEqual(self.bandit.cur_round, 1)
        self.assertEqual(self.bandit.cur_arm, 0)
        self.assertEqual(self.bandit.num_pulls, [0, 0, 0])
        self.assertEqual(self.bandit.active_arms, [0, 1, 2])
        self.assertEqual(self.bandit.cumulative_rewards, [0, 0, 0])

if __name__ == '__main__':
    unittest.main()