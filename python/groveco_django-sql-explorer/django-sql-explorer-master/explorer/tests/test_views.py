import json
import time

from django.test import TestCase
from django.core.urlresolvers import reverse
from django.contrib.auth.models import User
from django.conf import settings
from django.forms.models import model_to_dict

from explorer.tests.factories import SimpleQueryFactory, QueryLogFactory
from explorer.models import Query, QueryLog, MSG_FAILED_BLACKLIST
from explorer.utils import user_can_see_query
from explorer.app_settings import EXPLORER_TOKEN
from mock import Mock, patch


_ = lambda x: x


class TestQueryListView(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_admin_required(self):
        self.client.logout()
        resp = self.client.get(reverse("explorer_index"))
        self.assertTemplateUsed(resp, 'admin/login.html')

    def test_headers(self):
        SimpleQueryFactory(title='foo - bar1')
        SimpleQueryFactory(title='foo - bar2')
        SimpleQueryFactory(title='foo - bar3')
        SimpleQueryFactory(title='qux - mux')
        resp = self.client.get(reverse("explorer_index"))
        self.assertContains(resp, 'foo (3)')
        self.assertContains(resp, 'foo - bar2')
        self.assertContains(resp, 'qux - mux')

    def test_permissions_show_only_allowed_queries(self):
        self.client.logout()
        q1 = SimpleQueryFactory(title='canseethisone')
        q2 = SimpleQueryFactory(title='nope')
        user = User.objects.create_user('user', 'user@user.com', 'pwd')
        self.client.login(username='user', password='pwd')
        with self.settings(EXPLORER_USER_QUERY_VIEWS={user.id: [q1.id]}):
            resp = self.client.get(reverse("explorer_index"))
        self.assertTemplateUsed(resp, 'explorer/query_list.html')
        self.assertContains(resp, q1.title)
        self.assertNotContains(resp, q2.title)

    def test_run_count(self):
        q = SimpleQueryFactory(title='foo - bar1')
        for i in range(0, 4):
            q.log()
        resp = self.client.get(reverse("explorer_index"))
        self.assertContains(resp, '<td>4</td>')


class TestQueryCreateView(TestCase):

    def setUp(self):
        self.admin = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.user = User.objects.create_user('user', 'user@user.com', 'pwd')

    def test_change_permission_required(self):
        self.client.login(username='user', password='pwd')
        resp = self.client.get(reverse("query_create"))
        self.assertTemplateUsed(resp, 'admin/login.html')

    def test_renders_with_title(self):
        self.client.login(username='admin', password='pwd')
        resp = self.client.get(reverse("query_create"))
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertContains(resp, "New Query")


class TestQueryDetailView(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_query_with_bad_sql_renders_error(self):
        query = SimpleQueryFactory(sql="error")
        resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}))
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertContains(resp, "syntax error")

    def test_query_with_bad_sql_renders_error_on_save(self):
        query = SimpleQueryFactory(sql="select 1;")
        resp = self.client.post(reverse("query_detail", kwargs={'query_id': query.id}), data={'sql': 'error'})
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertContains(resp, "syntax error")

    def test_posting_query_saves_correctly(self):
        expected = 'select 2;'
        query = SimpleQueryFactory(sql="select 1;")
        data = model_to_dict(query)
        data['sql'] = expected
        self.client.post(reverse("query_detail", kwargs={'query_id': query.id}), data)
        self.assertEqual(Query.objects.get(pk=query.id).sql, expected)

    def test_change_permission_required_to_save_query(self):
        query = SimpleQueryFactory()
        expected = query.sql
        resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}))
        self.assertTemplateUsed(resp, 'explorer/query.html')

        self.client.post(reverse("query_detail", kwargs={'query_id': query.id}), {'sql': 'select 1;'})
        self.assertEqual(Query.objects.get(pk=query.id).sql, expected)

    def test_modified_date_gets_updated_after_viewing_query(self):
        query = SimpleQueryFactory()
        old = query.last_run_date
        time.sleep(0.1)
        self.client.get(reverse("query_detail", kwargs={'query_id': query.id}))
        self.assertNotEqual(old, Query.objects.get(pk=query.id).last_run_date)

    def test_doesnt_render_results_if_show_is_none(self):
        query = SimpleQueryFactory(sql='select 6870+1;')
        resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}) + '?show=0')
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertNotContains(resp, '6871')

    def test_doesnt_render_results_if_show_is_none_on_post(self):
        query = SimpleQueryFactory(sql='select 6870+1;')
        resp = self.client.post(reverse("query_detail", kwargs={'query_id': query.id}) + '?show=0', {'sql': 'select 6870+2;'})
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertNotContains(resp, '6872')

    def test_admin_required(self):
        self.client.logout()
        query = SimpleQueryFactory()
        resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}))
        self.assertTemplateUsed(resp, 'admin/login.html')

    def test_individual_view_permission(self):
        self.client.logout()
        user = User.objects.create_user('user1', 'user@user.com', 'pwd')
        self.client.login(username='user1', password='pwd')

        query = SimpleQueryFactory(sql="select 123+1")

        with self.settings(EXPLORER_USER_QUERY_VIEWS={user.id: [query.id]}):
            resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}))
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertContains(resp, "124")

    def test_token_auth(self):
        self.client.logout()

        query = SimpleQueryFactory(sql="select 123+1")

        with self.settings(EXPLORER_TOKEN_AUTH_ENABLED=True):
            resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}), **{'HTTP_X_API_TOKEN': EXPLORER_TOKEN})
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertContains(resp, "124")

    def test_token_auth(self):
        self.client.logout()

        query = SimpleQueryFactory(sql="select 123+1")

        with self.settings(EXPLORER_TOKEN_AUTH_ENABLED=True):
            resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}) + '?token=%s' % EXPLORER_TOKEN)
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertContains(resp, "124")

    def test_user_query_views(self):
        request = Mock()

        request.user.is_anonymous = Mock(return_value=True)
        kwargs = {}
        self.assertFalse(user_can_see_query(request, **kwargs))

        request.user.is_anonymous = Mock(return_value=True)
        self.assertFalse(user_can_see_query(request, **kwargs))

        kwargs = {'query_id': 123}
        request.user.is_anonymous = Mock(return_value=False)
        self.assertFalse(user_can_see_query(request, **kwargs))

        request.user.id = 99
        with self.settings(EXPLORER_USER_QUERY_VIEWS={99: [111, 123]}):
            self.assertTrue(user_can_see_query(request, **kwargs))

    @patch('explorer.models.get_s3_connection')
    def test_query_snapshot_renders(self, mocked_conn):
        conn = Mock()
        conn.list = Mock()
        conn.list.return_value = [{'key': 'foo-snapshot', 'last_modified': '2015-01-01'}
                                  ,{'key': 'bar-snapshot', 'last_modified': '2015-01-02'}]
        mocked_conn.return_value = conn
        query = SimpleQueryFactory(sql="select 1;", snapshot=True)
        resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}))
        self.assertContains(resp, '2015-01-01')
        self.assertContains(resp, '2015-01-02')
        self.assertContains(resp, settings.EXPLORER_S3_BUCKET)

    @patch('explorer.models.get_connection')
    def test_failing_blacklist_means_query_doesnt_execute(self, mocked_conn):
        # I should really learn to set up mocks correctly because this CANT be the most efficient way...
        cursor_result = Mock()
        cursor_result.fetchall.return_value = []
        cursor_result.description = [('foo', 'bar')]

        conn = Mock()
        conn.cursor.return_value = cursor_result
        mocked_conn.return_value = conn

        query = SimpleQueryFactory(sql="select 1;")
        resp = self.client.post(reverse("query_detail", kwargs={'query_id': query.id}), data={'sql': "select 'delete';"})
        self.assertTemplateUsed(resp, 'explorer/query.html')
        self.assertContains(resp, MSG_FAILED_BLACKLIST % '')

        # Feels fragile, but nor sure how else to access the called-with params of .execute
        self.assertEqual(conn.cursor.mock_calls[1][1][0], "select 1;")

    def test_fullscreen(self):
        query = SimpleQueryFactory(sql="select 1;")
        resp = self.client.get(reverse("query_detail", kwargs={'query_id': query.id}) + '?fullscreen=1')
        self.assertTemplateUsed(resp, 'explorer/fullscreen.html')


class TestDownloadView(TestCase):
    def setUp(self):
        self.query = SimpleQueryFactory(sql="select 1;")
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_admin_required(self):
        self.client.logout()
        resp = self.client.get(reverse("download_query", kwargs={'query_id': self.query.id}))
        self.assertTemplateUsed(resp, 'admin/login.html')

    def test_params_in_download(self):
        q = SimpleQueryFactory(sql="select '$$foo$$';")
        url = '%s?params=%s' % (reverse("download_query", kwargs={'query_id': q.id}), 'foo:123')
        resp = self.client.get(url)
        self.assertContains(resp, "'123'")

    def test_download_defaults_to_csv(self):
        query = SimpleQueryFactory()
        url = reverse("download_query", args=[query.pk])

        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['content-type'], 'text/csv')

    def test_download_csv(self):
        query = SimpleQueryFactory()
        url = reverse("download_query", args=[query.pk]) + '?format=csv'

        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['content-type'], 'text/csv')

    def test_download_json(self):
        query = SimpleQueryFactory()
        url = reverse("download_query", args=[query.pk]) + '?format=json'

        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['content-type'], 'application/json')

        json_data = json.loads(response.content.decode('utf-8'))
        self.assertIsInstance(json_data, list)
        self.assertEqual(len(json_data), 1)
        self.assertEqual(json_data, [{'TWO': 2}])


class TestQueryPlayground(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_empty_playground_renders(self):
        resp = self.client.get(reverse("explorer_playground"))
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, 'explorer/play.html')

    def test_playground_renders_with_query_sql(self):
        query = SimpleQueryFactory(sql="select 1;")
        resp = self.client.get('%s?query_id=%s' % (reverse("explorer_playground"), query.id))
        self.assertTemplateUsed(resp, 'explorer/play.html')
        self.assertContains(resp, 'select 1;')

    def test_playground_renders_with_posted_sql(self):
        resp = self.client.post(reverse("explorer_playground"), {'sql': 'select 1+3400;'})
        self.assertTemplateUsed(resp, 'explorer/play.html')
        self.assertContains(resp, '3401')

    def test_playground_doesnt_render_with_posted_sql_if_show_is_none(self):
        resp = self.client.post(reverse("explorer_playground") + '?show=0', {'sql': 'select 1+3400;'})
        self.assertTemplateUsed(resp, 'explorer/play.html')
        self.assertNotContains(resp, '3401')

    def test_playground_renders_with_empty_posted_sql(self):
        resp = self.client.post(reverse("explorer_playground"), {'sql': ''})
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, 'explorer/play.html')

    def test_query_with_no_resultset_doesnt_throw_error(self):
        query = SimpleQueryFactory(sql="")
        resp = self.client.get('%s?query_id=%s' % (reverse("explorer_playground"), query.id))
        self.assertTemplateUsed(resp, 'explorer/play.html')

    def test_admin_required(self):
        self.client.logout()
        resp = self.client.get(reverse("explorer_playground"))
        self.assertTemplateUsed(resp, 'admin/login.html')

    def test_loads_query_from_log(self):
        querylog = QueryLogFactory()
        resp = self.client.get('%s?querylog_id=%s' % (reverse("explorer_playground"), querylog.id))
        self.assertContains(resp, "FOUR")

    def test_fails_blacklist(self):
        resp = self.client.post(reverse("explorer_playground"), {'sql': "select 'delete'"})
        self.assertTemplateUsed(resp, 'explorer/play.html')
        self.assertContains(resp, MSG_FAILED_BLACKLIST % '')

    def test_fullscreen(self):
        query = SimpleQueryFactory(sql="")
        resp = self.client.get('%s?query_id=%s&fullscreen=1' % (reverse("explorer_playground"), query.id))
        self.assertTemplateUsed(resp, 'explorer/fullscreen.html')


class TestCSVFromSQL(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_admin_required(self):
        self.client.logout()
        resp = self.client.post(reverse("download_sql"), {})
        self.assertTemplateUsed(resp, 'admin/login.html')

    def test_downloading_from_playground(self):
        sql = "select 1;"
        resp = self.client.post(reverse("download_sql"), {'sql': sql})
        self.assertIn('attachment', resp['Content-Disposition'])
        self.assertEqual('text/csv', resp['content-type'])
        ql = QueryLog.objects.first()
        self.assertIn('filename="Playground_-_%s.csv"' % ql.id, resp['Content-Disposition'])

    def test_stream_csv_from_query(self):
        q = SimpleQueryFactory()
        resp = self.client.get(reverse("stream_query", kwargs={'query_id': q.id}))
        self.assertEqual('text/csv', resp['content-type'])


class TestSQLDownloadViews(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_sql_download_csv(self):
        url = reverse("download_sql") + '?format=csv'

        response = self.client.post(url, {'sql': 'select 1;'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['content-type'], 'text/csv')

    def test_sql_download_csv_with_custom_delim(self):
        url = reverse("download_sql") + '?format=csv&delim=|'

        response = self.client.post(url, {'sql': 'select 1,2;'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['content-type'], 'text/csv')
        self.assertEqual(response.content.decode('utf-8'), '1|2\r\n1|2\r\n')

    def test_sql_download_csv_with_tab_delim(self):
        url = reverse("download_sql") + '?format=csv&delim=tab'

        response = self.client.post(url, {'sql': 'select 1,2;'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['content-type'], 'text/csv')
        self.assertEqual(response.content.decode('utf-8'), '1\t2\r\n1\t2\r\n')

    def test_sql_download_csv_with_bad_delim(self):
        url = reverse("download_sql") + '?format=csv&delim=foo'

        response = self.client.post(url, {'sql': 'select 1,2;'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['content-type'], 'text/csv')
        self.assertEqual(response.content.decode('utf-8'), '1,2\r\n1,2\r\n')

    def test_sql_download_json(self):
        url = reverse("download_sql") + '?format=json'

        response = self.client.post(url, {'sql': 'select 1;'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['content-type'], 'application/json')


class TestSchemaView(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_returns_schema_contents(self):
        resp = self.client.get(reverse("explorer_schema"))
        self.assertContains(resp, "explorer_query")
        self.assertTemplateUsed(resp, 'explorer/schema.html')

    def test_admin_required(self):
        self.client.logout()
        resp = self.client.get(reverse("explorer_schema"))
        self.assertTemplateUsed(resp, 'admin/login.html')


class TestFormat(TestCase):
    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_returns_formatted_sql(self):
        resp = self.client.post(reverse("format_sql"),  data={"sql": "select * from explorer_query"})
        resp = json.loads(resp.content.decode('utf-8'))
        self.assertIn("\n", resp['formatted'])
        self.assertIn("explorer_query", resp['formatted'])


class TestParamsInViews(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')
        self.query = SimpleQueryFactory(sql="select $$swap$$;")

    def test_retrieving_query_works_with_params(self):
        resp = self.client.get(reverse("query_detail", kwargs={'query_id': self.query.id}) + '?params=swap:123}')
        self.assertContains(resp, "123")

    def test_saving_non_executing_query_with__wrong_url_params_works(self):
        q = SimpleQueryFactory(sql="select $$swap$$;")
        data = model_to_dict(q)
        url = '%s?params=%s' % (reverse("query_detail", kwargs={'query_id': q.id}), 'foo:123')
        resp = self.client.post(url, data)
        self.assertContains(resp, 'saved')

    def test_users_without_change_permissions_can_use_params(self):
        resp = self.client.get(reverse("query_detail", kwargs={'query_id': self.query.id}) + '?params=swap:123}')
        self.assertContains(resp, "123")


class TestCreatedBy(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.user2 = User.objects.create_superuser('admin2', 'admin2@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')
        self.query = SimpleQueryFactory.build()
        self.data = model_to_dict(self.query)
        self.data["created_by_user"] = 2

    def test_query_update_doesnt_change_created_user(self):
        self.query.save()
        self.client.post(reverse("query_detail", kwargs={'query_id': self.query.id}), self.data)
        q = Query.objects.get(id=self.query.id)
        self.assertEqual(q.created_by_user_id, 1)

    def test_new_query_gets_created_by_logged_in_user(self):
        self.client.post(reverse("query_create"), self.data)
        q = Query.objects.first()
        self.assertEqual(q.created_by_user_id, 1)


class TestQueryLog(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    def test_playground_saves_query_to_log(self):
        self.client.post(reverse("explorer_playground"), {'sql': 'select 1;'})
        log = QueryLog.objects.first()
        self.assertTrue(log.is_playground)
        self.assertEqual(log.sql, 'select 1;')

    # Since it will be saved on the initial query creation, no need to log it
    def test_creating_query_does_not_save_to_log(self):
        query = SimpleQueryFactory()
        self.client.post(reverse("query_create"), model_to_dict(query))
        self.assertEqual(0, QueryLog.objects.count())

    def test_query_saves_to_log(self):
        query = SimpleQueryFactory()
        data = model_to_dict(query)
        data['sql'] = 'select 12345;'
        self.client.post(reverse("query_detail", kwargs={'query_id': query.id}), data)
        self.assertEqual(1, QueryLog.objects.count())

    def test_query_gets_logged_and_appears_on_log_page(self):
        query = SimpleQueryFactory()
        data = model_to_dict(query)
        data['sql'] = 'select 12345;'
        self.client.post(reverse("query_detail", kwargs={'query_id': query.id}), data)
        resp = self.client.get(reverse("explorer_logs"))
        self.assertContains(resp, 'select 12345;')

    def test_admin_required(self):
        self.client.logout()
        resp = self.client.get(reverse("explorer_logs"))
        self.assertTemplateUsed(resp, 'admin/login.html')

    def test_is_playground(self):
        self.assertTrue(QueryLog(sql='foo').is_playground)

        q = SimpleQueryFactory()
        self.assertFalse(QueryLog(sql='foo', query_id=q.id).is_playground)


class TestEmailQuery(TestCase):

    def setUp(self):
        self.user = User.objects.create_superuser('admin', 'admin@admin.com', 'pwd')
        self.client.login(username='admin', password='pwd')

    @patch('explorer.views.execute_query')
    def test_email_calls_task(self, mocked_execute):
        query = SimpleQueryFactory()
        url = reverse("email_csv_query", kwargs={'query_id': query.id})
        self.client.post(url, data={'email': 'foo@bar.com'}, **{'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'})
        self.assertEqual(mocked_execute.delay.call_count, 1)
