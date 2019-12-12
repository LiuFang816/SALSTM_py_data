# -*- coding: utf-8 -*-
# Copyright 2015, 2016 OpenMarket Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tempfile
from synapse.config._base import ConfigError
from tests import unittest
from twisted.internet import defer

from tests.utils import setup_test_homeserver
from synapse.appservice import ApplicationService, ApplicationServiceState
from synapse.storage.appservice import (
    ApplicationServiceStore, ApplicationServiceTransactionStore
)

import json
import os
import yaml
from mock import Mock


class ApplicationServiceStoreTestCase(unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        self.as_yaml_files = []
        config = Mock(
            app_service_config_files=self.as_yaml_files,
            event_cache_size=1,
            password_providers=[],
        )
        hs = yield setup_test_homeserver(
            config=config,
            federation_sender=Mock(),
            replication_layer=Mock(),
        )

        self.as_token = "token1"
        self.as_url = "some_url"
        self.as_id = "as1"
        self._add_appservice(
            self.as_token,
            self.as_id,
            self.as_url,
            "some_hs_token",
            "bob"
        )
        self._add_appservice("token2", "as2", "some_url", "some_hs_token", "bob")
        self._add_appservice("token3", "as3", "some_url", "some_hs_token", "bob")
        # must be done after inserts
        self.store = ApplicationServiceStore(hs)

    def tearDown(self):
        # TODO: suboptimal that we need to create files for tests!
        for f in self.as_yaml_files:
            try:
                os.remove(f)
            except:
                pass

    def _add_appservice(self, as_token, id, url, hs_token, sender):
        as_yaml = dict(url=url, as_token=as_token, hs_token=hs_token,
                       id=id, sender_localpart=sender, namespaces={})
        # use the token as the filename
        with open(as_token, 'w') as outfile:
            outfile.write(yaml.dump(as_yaml))
            self.as_yaml_files.append(as_token)

    def test_retrieve_unknown_service_token(self):
        service = self.store.get_app_service_by_token("invalid_token")
        self.assertEquals(service, None)

    def test_retrieval_of_service(self):
        stored_service = self.store.get_app_service_by_token(
            self.as_token
        )
        self.assertEquals(stored_service.token, self.as_token)
        self.assertEquals(stored_service.id, self.as_id)
        self.assertEquals(stored_service.url, self.as_url)
        self.assertEquals(
            stored_service.namespaces[ApplicationService.NS_ALIASES],
            []
        )
        self.assertEquals(
            stored_service.namespaces[ApplicationService.NS_ROOMS],
            []
        )
        self.assertEquals(
            stored_service.namespaces[ApplicationService.NS_USERS],
            []
        )

    def test_retrieval_of_all_services(self):
        services = self.store.get_app_services()
        self.assertEquals(len(services), 3)


class ApplicationServiceTransactionStoreTestCase(unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        self.as_yaml_files = []

        config = Mock(
            app_service_config_files=self.as_yaml_files,
            event_cache_size=1,
            password_providers=[],
        )
        hs = yield setup_test_homeserver(
            config=config,
            federation_sender=Mock(),
            replication_layer=Mock(),
        )
        self.db_pool = hs.get_db_pool()

        self.as_list = [
            {
                "token": "token1",
                "url": "https://matrix-as.org",
                "id": "id_1"
            },
            {
                "token": "alpha_tok",
                "url": "https://alpha.com",
                "id": "id_alpha"
            },
            {
                "token": "beta_tok",
                "url": "https://beta.com",
                "id": "id_beta"
            },
            {
                "token": "gamma_tok",
                "url": "https://gamma.com",
                "id": "id_gamma"
            },
        ]
        for s in self.as_list:
            yield self._add_service(s["url"], s["token"], s["id"])

        self.as_yaml_files = []

        self.store = TestTransactionStore(hs)

    def _add_service(self, url, as_token, id):
        as_yaml = dict(url=url, as_token=as_token, hs_token="something",
                       id=id, sender_localpart="a_sender", namespaces={})
        # use the token as the filename
        with open(as_token, 'w') as outfile:
            outfile.write(yaml.dump(as_yaml))
            self.as_yaml_files.append(as_token)

    def _set_state(self, id, state, txn=None):
        return self.db_pool.runQuery(
            "INSERT INTO application_services_state(as_id, state, last_txn) "
            "VALUES(?,?,?)",
            (id, state, txn)
        )

    def _insert_txn(self, as_id, txn_id, events):
        return self.db_pool.runQuery(
            "INSERT INTO application_services_txns(as_id, txn_id, event_ids) "
            "VALUES(?,?,?)",
            (as_id, txn_id, json.dumps([e.event_id for e in events]))
        )

    def _set_last_txn(self, as_id, txn_id):
        return self.db_pool.runQuery(
            "INSERT INTO application_services_state(as_id, last_txn, state) "
            "VALUES(?,?,?)",
            (as_id, txn_id, ApplicationServiceState.UP)
        )

    @defer.inlineCallbacks
    def test_get_appservice_state_none(self):
        service = Mock(id=999)
        state = yield self.store.get_appservice_state(service)
        self.assertEquals(None, state)

    @defer.inlineCallbacks
    def test_get_appservice_state_up(self):
        yield self._set_state(
            self.as_list[0]["id"], ApplicationServiceState.UP
        )
        service = Mock(id=self.as_list[0]["id"])
        state = yield self.store.get_appservice_state(service)
        self.assertEquals(ApplicationServiceState.UP, state)

    @defer.inlineCallbacks
    def test_get_appservice_state_down(self):
        yield self._set_state(
            self.as_list[0]["id"], ApplicationServiceState.UP
        )
        yield self._set_state(
            self.as_list[1]["id"], ApplicationServiceState.DOWN
        )
        yield self._set_state(
            self.as_list[2]["id"], ApplicationServiceState.DOWN
        )
        service = Mock(id=self.as_list[1]["id"])
        state = yield self.store.get_appservice_state(service)
        self.assertEquals(ApplicationServiceState.DOWN, state)

    @defer.inlineCallbacks
    def test_get_appservices_by_state_none(self):
        services = yield self.store.get_appservices_by_state(
            ApplicationServiceState.DOWN
        )
        self.assertEquals(0, len(services))

    @defer.inlineCallbacks
    def test_set_appservices_state_down(self):
        service = Mock(id=self.as_list[1]["id"])
        yield self.store.set_appservice_state(
            service,
            ApplicationServiceState.DOWN
        )
        rows = yield self.db_pool.runQuery(
            "SELECT as_id FROM application_services_state WHERE state=?",
            (ApplicationServiceState.DOWN,)
        )
        self.assertEquals(service.id, rows[0][0])

    @defer.inlineCallbacks
    def test_set_appservices_state_multiple_up(self):
        service = Mock(id=self.as_list[1]["id"])
        yield self.store.set_appservice_state(
            service,
            ApplicationServiceState.UP
        )
        yield self.store.set_appservice_state(
            service,
            ApplicationServiceState.DOWN
        )
        yield self.store.set_appservice_state(
            service,
            ApplicationServiceState.UP
        )
        rows = yield self.db_pool.runQuery(
            "SELECT as_id FROM application_services_state WHERE state=?",
            (ApplicationServiceState.UP,)
        )
        self.assertEquals(service.id, rows[0][0])

    @defer.inlineCallbacks
    def test_create_appservice_txn_first(self):
        service = Mock(id=self.as_list[0]["id"])
        events = [Mock(event_id="e1"), Mock(event_id="e2")]
        txn = yield self.store.create_appservice_txn(service, events)
        self.assertEquals(txn.id, 1)
        self.assertEquals(txn.events, events)
        self.assertEquals(txn.service, service)

    @defer.inlineCallbacks
    def test_create_appservice_txn_older_last_txn(self):
        service = Mock(id=self.as_list[0]["id"])
        events = [Mock(event_id="e1"), Mock(event_id="e2")]
        yield self._set_last_txn(service.id, 9643)  # AS is falling behind
        yield self._insert_txn(service.id, 9644, events)
        yield self._insert_txn(service.id, 9645, events)
        txn = yield self.store.create_appservice_txn(service, events)
        self.assertEquals(txn.id, 9646)
        self.assertEquals(txn.events, events)
        self.assertEquals(txn.service, service)

    @defer.inlineCallbacks
    def test_create_appservice_txn_up_to_date_last_txn(self):
        service = Mock(id=self.as_list[0]["id"])
        events = [Mock(event_id="e1"), Mock(event_id="e2")]
        yield self._set_last_txn(service.id, 9643)
        txn = yield self.store.create_appservice_txn(service, events)
        self.assertEquals(txn.id, 9644)
        self.assertEquals(txn.events, events)
        self.assertEquals(txn.service, service)

    @defer.inlineCallbacks
    def test_create_appservice_txn_up_fuzzing(self):
        service = Mock(id=self.as_list[0]["id"])
        events = [Mock(event_id="e1"), Mock(event_id="e2")]
        yield self._set_last_txn(service.id, 9643)

        # dump in rows with higher IDs to make sure the queries aren't wrong.
        yield self._set_last_txn(self.as_list[1]["id"], 119643)
        yield self._set_last_txn(self.as_list[2]["id"], 9)
        yield self._set_last_txn(self.as_list[3]["id"], 9643)
        yield self._insert_txn(self.as_list[1]["id"], 119644, events)
        yield self._insert_txn(self.as_list[1]["id"], 119645, events)
        yield self._insert_txn(self.as_list[1]["id"], 119646, events)
        yield self._insert_txn(self.as_list[2]["id"], 10, events)
        yield self._insert_txn(self.as_list[3]["id"], 9643, events)

        txn = yield self.store.create_appservice_txn(service, events)
        self.assertEquals(txn.id, 9644)
        self.assertEquals(txn.events, events)
        self.assertEquals(txn.service, service)

    @defer.inlineCallbacks
    def test_complete_appservice_txn_first_txn(self):
        service = Mock(id=self.as_list[0]["id"])
        events = [Mock(event_id="e1"), Mock(event_id="e2")]
        txn_id = 1

        yield self._insert_txn(service.id, txn_id, events)
        yield self.store.complete_appservice_txn(txn_id=txn_id, service=service)

        res = yield self.db_pool.runQuery(
            "SELECT last_txn FROM application_services_state WHERE as_id=?",
            (service.id,)
        )
        self.assertEquals(1, len(res))
        self.assertEquals(txn_id, res[0][0])

        res = yield self.db_pool.runQuery(
            "SELECT * FROM application_services_txns WHERE txn_id=?",
            (txn_id,)
        )
        self.assertEquals(0, len(res))

    @defer.inlineCallbacks
    def test_complete_appservice_txn_existing_in_state_table(self):
        service = Mock(id=self.as_list[0]["id"])
        events = [Mock(event_id="e1"), Mock(event_id="e2")]
        txn_id = 5
        yield self._set_last_txn(service.id, 4)
        yield self._insert_txn(service.id, txn_id, events)
        yield self.store.complete_appservice_txn(txn_id=txn_id, service=service)

        res = yield self.db_pool.runQuery(
            "SELECT last_txn, state FROM application_services_state WHERE "
            "as_id=?",
            (service.id,)
        )
        self.assertEquals(1, len(res))
        self.assertEquals(txn_id, res[0][0])
        self.assertEquals(ApplicationServiceState.UP, res[0][1])

        res = yield self.db_pool.runQuery(
            "SELECT * FROM application_services_txns WHERE txn_id=?",
            (txn_id,)
        )
        self.assertEquals(0, len(res))

    @defer.inlineCallbacks
    def test_get_oldest_unsent_txn_none(self):
        service = Mock(id=self.as_list[0]["id"])

        txn = yield self.store.get_oldest_unsent_txn(service)
        self.assertEquals(None, txn)

    @defer.inlineCallbacks
    def test_get_oldest_unsent_txn(self):
        service = Mock(id=self.as_list[0]["id"])
        events = [Mock(event_id="e1"), Mock(event_id="e2")]
        other_events = [Mock(event_id="e5"), Mock(event_id="e6")]

        # we aren't testing store._base stuff here, so mock this out
        self.store._get_events = Mock(return_value=events)

        yield self._insert_txn(self.as_list[1]["id"], 9, other_events)
        yield self._insert_txn(service.id, 10, events)
        yield self._insert_txn(service.id, 11, other_events)
        yield self._insert_txn(service.id, 12, other_events)

        txn = yield self.store.get_oldest_unsent_txn(service)
        self.assertEquals(service, txn.service)
        self.assertEquals(10, txn.id)
        self.assertEquals(events, txn.events)

    @defer.inlineCallbacks
    def test_get_appservices_by_state_single(self):
        yield self._set_state(
            self.as_list[0]["id"], ApplicationServiceState.DOWN
        )
        yield self._set_state(
            self.as_list[1]["id"], ApplicationServiceState.UP
        )

        services = yield self.store.get_appservices_by_state(
            ApplicationServiceState.DOWN
        )
        self.assertEquals(1, len(services))
        self.assertEquals(self.as_list[0]["id"], services[0].id)

    @defer.inlineCallbacks
    def test_get_appservices_by_state_multiple(self):
        yield self._set_state(
            self.as_list[0]["id"], ApplicationServiceState.DOWN
        )
        yield self._set_state(
            self.as_list[1]["id"], ApplicationServiceState.UP
        )
        yield self._set_state(
            self.as_list[2]["id"], ApplicationServiceState.DOWN
        )
        yield self._set_state(
            self.as_list[3]["id"], ApplicationServiceState.UP
        )

        services = yield self.store.get_appservices_by_state(
            ApplicationServiceState.DOWN
        )
        self.assertEquals(2, len(services))
        self.assertEquals(
            set([self.as_list[2]["id"], self.as_list[0]["id"]]),
            set([services[0].id, services[1].id])
        )


# required for ApplicationServiceTransactionStoreTestCase tests
class TestTransactionStore(ApplicationServiceTransactionStore,
                           ApplicationServiceStore):

    def __init__(self, hs):
        super(TestTransactionStore, self).__init__(hs)


class ApplicationServiceStoreConfigTestCase(unittest.TestCase):

    def _write_config(self, suffix, **kwargs):
        vals = {
            "id": "id" + suffix,
            "url": "url" + suffix,
            "as_token": "as_token" + suffix,
            "hs_token": "hs_token" + suffix,
            "sender_localpart": "sender_localpart" + suffix,
            "namespaces": {},
        }
        vals.update(kwargs)

        _, path = tempfile.mkstemp(prefix="as_config")
        with open(path, "w") as f:
            f.write(yaml.dump(vals))
        return path

    @defer.inlineCallbacks
    def test_unique_works(self):
        f1 = self._write_config(suffix="1")
        f2 = self._write_config(suffix="2")

        config = Mock(
            app_service_config_files=[f1, f2], event_cache_size=1,
            password_providers=[]
        )
        hs = yield setup_test_homeserver(
            config=config,
            datastore=Mock(),
            federation_sender=Mock(),
            replication_layer=Mock(),
        )

        ApplicationServiceStore(hs)

    @defer.inlineCallbacks
    def test_duplicate_ids(self):
        f1 = self._write_config(id="id", suffix="1")
        f2 = self._write_config(id="id", suffix="2")

        config = Mock(
            app_service_config_files=[f1, f2], event_cache_size=1,
            password_providers=[]
        )
        hs = yield setup_test_homeserver(
            config=config,
            datastore=Mock(),
            federation_sender=Mock(),
            replication_layer=Mock(),
        )

        with self.assertRaises(ConfigError) as cm:
            ApplicationServiceStore(hs)

        e = cm.exception
        self.assertIn(f1, e.message)
        self.assertIn(f2, e.message)
        self.assertIn("id", e.message)

    @defer.inlineCallbacks
    def test_duplicate_as_tokens(self):
        f1 = self._write_config(as_token="as_token", suffix="1")
        f2 = self._write_config(as_token="as_token", suffix="2")

        config = Mock(
            app_service_config_files=[f1, f2], event_cache_size=1,
            password_providers=[]
        )
        hs = yield setup_test_homeserver(
            config=config,
            datastore=Mock(),
            federation_sender=Mock(),
            replication_layer=Mock(),
        )

        with self.assertRaises(ConfigError) as cm:
            ApplicationServiceStore(hs)

        e = cm.exception
        self.assertIn(f1, e.message)
        self.assertIn(f2, e.message)
        self.assertIn("as_token", e.message)
