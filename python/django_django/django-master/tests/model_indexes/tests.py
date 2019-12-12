from django.db import models
from django.test import SimpleTestCase

from .models import Book


class IndexesTests(SimpleTestCase):

    def test_suffix(self):
        self.assertEqual(models.Index.suffix, 'idx')

    def test_repr(self):
        index = models.Index(fields=['title'])
        multi_col_index = models.Index(fields=['title', 'author'])
        self.assertEqual(repr(index), "<Index: fields='title'>")
        self.assertEqual(repr(multi_col_index), "<Index: fields='title, author'>")

    def test_eq(self):
        index = models.Index(fields=['title'])
        same_index = models.Index(fields=['title'])
        another_index = models.Index(fields=['title', 'author'])
        index.model = Book
        same_index.model = Book
        another_index.model = Book
        self.assertEqual(index, same_index)
        self.assertNotEqual(index, another_index)

    def test_index_fields_type(self):
        with self.assertRaisesMessage(ValueError, 'Index.fields must be a list.'):
            models.Index(fields='title')

    def test_raises_error_without_field(self):
        msg = 'At least one field is required to define an index.'
        with self.assertRaisesMessage(ValueError, msg):
            models.Index()

    def test_max_name_length(self):
        msg = 'Index names cannot be longer than 30 characters.'
        with self.assertRaisesMessage(ValueError, msg):
            models.Index(fields=['title'], name='looooooooooooong_index_name_idx')

    def test_name_constraints(self):
        msg = 'Index names cannot start with an underscore (_).'
        with self.assertRaisesMessage(ValueError, msg):
            models.Index(fields=['title'], name='_name_starting_with_underscore')

        msg = 'Index names cannot start with a number (0-9).'
        with self.assertRaisesMessage(ValueError, msg):
            models.Index(fields=['title'], name='5name_starting_with_number')

    def test_name_auto_generation(self):
        index = models.Index(fields=['author'])
        index.set_name_with_model(Book)
        self.assertEqual(index.name, 'model_index_author_0f5565_idx')

        # '-' for DESC columns should be accounted for in the index name.
        index = models.Index(fields=['-author'])
        index.set_name_with_model(Book)
        self.assertEqual(index.name, 'model_index_author_708765_idx')

        # fields may be truncated in the name. db_column is used for naming.
        long_field_index = models.Index(fields=['pages'])
        long_field_index.set_name_with_model(Book)
        self.assertEqual(long_field_index.name, 'model_index_page_co_69235a_idx')

        # suffix can't be longer than 3 characters.
        long_field_index.suffix = 'suff'
        msg = 'Index too long for multiple database support. Is self.suffix longer than 3 characters?'
        with self.assertRaisesMessage(AssertionError, msg):
            long_field_index.set_name_with_model(Book)

    def test_deconstruction(self):
        index = models.Index(fields=['title'])
        index.set_name_with_model(Book)
        path, args, kwargs = index.deconstruct()
        self.assertEqual(path, 'django.db.models.Index')
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {'fields': ['title'], 'name': 'model_index_title_196f42_idx'})
