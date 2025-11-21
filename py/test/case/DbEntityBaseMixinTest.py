from test.autotest_common import *


class DbEntityBaseMixinTest(unittest.TestCase):

    def setUp(self):
        # Create an in-memory SQLite database for testing

        # Create a proper base for our test entity
        self.db_manager = DbConnectionManager()
        DbBase = self.db_manager.DeclDbBase()

        class TestEntity(DbEntityBaseMixin, DbBase):
            # A test entity that inherits from DbEntityBaseMixin
            __tablename__ = "test_entity"

            name = sqlalchemy.Column(sqlalchemy.String(50))
            value = sqlalchemy.Column(sqlalchemy.Integer)

        self.TestEntity = TestEntity

        self.SessionMaker = self.db_manager.connect("sqlite:///:memory:", echo=False)

    def tearDown(self):
        # Clean up database connections after tests
        if hasattr(self, "db_manager") and hasattr(self.db_manager, "engine"):
            self.db_manager.engine.dispose()

    def test_init_method(self):
        with self.SessionMaker() as session:
            # Test that init() sets id and create_time
            entity = self.TestEntity()
            entity.init()

            self.assertIsNotNone(entity.id)
            self.assertIsNotNone(entity.create_time)
            self.assertEqual(len(entity.id), 32)  # UUID hex string length

    def test_nowtime_method(self):
        # Test that nowtime() returns correctly formatted time
        time_str = self.TestEntity.nowtime()
        self.assertRegex(time_str, r"^\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}$")

    def test_added_method(self):
        with self.SessionMaker() as session:
            # Test that added() initializes and adds entity to session
            entity = self.TestEntity(name="Test", value=42)
            returned_entity = entity.added(session)

            # Check that the entity was initialized
            self.assertIsNotNone(entity.id)
            self.assertIsNotNone(entity.create_time)

            # Check that the returned entity is the same
            self.assertIs(returned_entity, entity)

            # Check that entity was added to session
            self.assertIn(entity, session.new)

    def test_to_dict_method(self):
        with self.SessionMaker() as session:
            # Test that to_dict() correctly converts entity to dictionary
            entity = self.TestEntity(name="Test", value=42)
            entity.id = "test-id"
            entity.create_time = "2023-01-01-12:00:00"

            result = entity.to_dict()

            expected = {
                "id": "test-id",
                "create_time": "2023-01-01-12:00:00",
                "name": "Test",
                "value": 42,
            }

            self.assertEqual(result, expected)

    def test_from_dict_method(self):
        with self.SessionMaker() as session:
            # Test that from_dict() correctly creates entity from dictionary
            data = {
                "id": "dict-id",
                "create_time": "2023-01-01-12:00:00",
                "name": "DictTest",
                "value": 100,
            }

            entity = self.TestEntity.from_dict(data)

            self.assertEqual(entity.id, "dict-id")
            self.assertEqual(entity.create_time, "2023-01-01-12:00:00")
            self.assertEqual(entity.name, "DictTest")
            self.assertEqual(entity.value, 100)

    def test_from_dict_with_move_from(self):
        with self.SessionMaker() as session:
            # Test that from_dict() correctly updates existing entity when move_from is provided
            existing_entity = self.TestEntity()
            existing_entity.id = "existing-id"
            existing_entity.create_time = "2022-01-01-12:00:00"
            existing_entity.name = "Existing"
            existing_entity.value = 50

            data = {"name": "UpdatedName", "value": 99}

            updated_entity = self.TestEntity.from_dict(data, move_from=existing_entity)

            # Should be the same object
            self.assertIs(updated_entity, existing_entity)

            # Fields present in data should be updated
            self.assertEqual(updated_entity.name, "UpdatedName")
            self.assertEqual(updated_entity.value, 99)

            # Fields not in data should retain original values
            self.assertEqual(updated_entity.id, "existing-id")
            self.assertEqual(updated_entity.create_time, "2022-01-01-12:00:00")
