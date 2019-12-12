from .test_base_class import ZhihuClientClassTest

PEOPLE_SLUG = 'giantchen'


class TestPeopleBadgeNumber(ZhihuClientClassTest):
    def test_badge_topics_number(self):
        self.assertEqual(
            len(list(self.client.people(PEOPLE_SLUG).badge.topics)), 3
        )

    def test_people_has_badge(self):
        self.assertTrue(self.client.people(PEOPLE_SLUG).badge.has_badge)

    def test_people_has_identity(self):
        self.assertFalse(self.client.people(PEOPLE_SLUG).badge.has_identity)

    def test_people_is_best_answerer_or_not(self):
        self.assertTrue(self.client.people(PEOPLE_SLUG).badge.is_best_answerer)

    def test_people_identify_information(self):
        self.assertIsNone(self.client.people(PEOPLE_SLUG).badge.identity)
