# -*- coding: utf-8 -*-
# Copyright 2014-2016 OpenMarket Ltd
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

from ._base import SQLBaseStore
from . import engines

from twisted.internet import defer

import ujson as json
import logging

logger = logging.getLogger(__name__)


class BackgroundUpdatePerformance(object):
    """Tracks the how long a background update is taking to update its items"""

    def __init__(self, name):
        self.name = name
        self.total_item_count = 0
        self.total_duration_ms = 0
        self.avg_item_count = 0
        self.avg_duration_ms = 0

    def update(self, item_count, duration_ms):
        """Update the stats after doing an update"""
        self.total_item_count += item_count
        self.total_duration_ms += duration_ms

        # Exponential moving averages for the number of items updated and
        # the duration.
        self.avg_item_count += 0.1 * (item_count - self.avg_item_count)
        self.avg_duration_ms += 0.1 * (duration_ms - self.avg_duration_ms)

    def average_items_per_ms(self):
        """An estimate of how long it takes to do a single update.
        Returns:
            A duration in ms as a float
        """
        if self.total_item_count == 0:
            return None
        else:
            # Use the exponential moving average so that we can adapt to
            # changes in how long the update process takes.
            return float(self.avg_item_count) / float(self.avg_duration_ms)

    def total_items_per_ms(self):
        """An estimate of how long it takes to do a single update.
        Returns:
            A duration in ms as a float
        """
        if self.total_item_count == 0:
            return None
        else:
            return float(self.total_item_count) / float(self.total_duration_ms)


class BackgroundUpdateStore(SQLBaseStore):
    """ Background updates are updates to the database that run in the
    background. Each update processes a batch of data at once. We attempt to
    limit the impact of each update by monitoring how long each batch takes to
    process and autotuning the batch size.
    """

    MINIMUM_BACKGROUND_BATCH_SIZE = 100
    DEFAULT_BACKGROUND_BATCH_SIZE = 100
    BACKGROUND_UPDATE_INTERVAL_MS = 1000
    BACKGROUND_UPDATE_DURATION_MS = 100

    def __init__(self, hs):
        super(BackgroundUpdateStore, self).__init__(hs)
        self._background_update_performance = {}
        self._background_update_queue = []
        self._background_update_handlers = {}
        self._background_update_timer = None

    @defer.inlineCallbacks
    def start_doing_background_updates(self):
        assert self._background_update_timer is None, \
            "background updates already running"

        logger.info("Starting background schema updates")

        while True:
            sleep = defer.Deferred()
            self._background_update_timer = self._clock.call_later(
                self.BACKGROUND_UPDATE_INTERVAL_MS / 1000., sleep.callback, None
            )
            try:
                yield sleep
            finally:
                self._background_update_timer = None

            try:
                result = yield self.do_next_background_update(
                    self.BACKGROUND_UPDATE_DURATION_MS
                )
            except:
                logger.exception("Error doing update")
            else:
                if result is None:
                    logger.info(
                        "No more background updates to do."
                        " Unscheduling background update task."
                    )
                    defer.returnValue(None)

    @defer.inlineCallbacks
    def do_next_background_update(self, desired_duration_ms):
        """Does some amount of work on the next queued background update

        Args:
            desired_duration_ms(float): How long we want to spend
                updating.
        Returns:
            A deferred that completes once some amount of work is done.
            The deferred will have a value of None if there is currently
            no more work to do.
        """
        if not self._background_update_queue:
            updates = yield self._simple_select_list(
                "background_updates",
                keyvalues=None,
                retcols=("update_name", "depends_on"),
            )
            in_flight = set(update["update_name"] for update in updates)
            for update in updates:
                if update["depends_on"] not in in_flight:
                    self._background_update_queue.append(update['update_name'])

        if not self._background_update_queue:
            # no work left to do
            defer.returnValue(None)

        # pop from the front, and add back to the back
        update_name = self._background_update_queue.pop(0)
        self._background_update_queue.append(update_name)

        res = yield self._do_background_update(update_name, desired_duration_ms)
        defer.returnValue(res)

    @defer.inlineCallbacks
    def _do_background_update(self, update_name, desired_duration_ms):
        logger.info("Starting update batch on background update '%s'",
                    update_name)

        update_handler = self._background_update_handlers[update_name]

        performance = self._background_update_performance.get(update_name)

        if performance is None:
            performance = BackgroundUpdatePerformance(update_name)
            self._background_update_performance[update_name] = performance

        items_per_ms = performance.average_items_per_ms()

        if items_per_ms is not None:
            batch_size = int(desired_duration_ms * items_per_ms)
            # Clamp the batch size so that we always make progress
            batch_size = max(batch_size, self.MINIMUM_BACKGROUND_BATCH_SIZE)
        else:
            batch_size = self.DEFAULT_BACKGROUND_BATCH_SIZE

        progress_json = yield self._simple_select_one_onecol(
            "background_updates",
            keyvalues={"update_name": update_name},
            retcol="progress_json"
        )

        progress = json.loads(progress_json)

        time_start = self._clock.time_msec()
        items_updated = yield update_handler(progress, batch_size)
        time_stop = self._clock.time_msec()

        duration_ms = time_stop - time_start

        logger.info(
            "Updating %r. Updated %r items in %rms."
            " (total_rate=%r/ms, current_rate=%r/ms, total_updated=%r, batch_size=%r)",
            update_name, items_updated, duration_ms,
            performance.total_items_per_ms(),
            performance.average_items_per_ms(),
            performance.total_item_count,
            batch_size,
        )

        performance.update(items_updated, duration_ms)

        defer.returnValue(len(self._background_update_performance))

    def register_background_update_handler(self, update_name, update_handler):
        """Register a handler for doing a background update.

        The handler should take two arguments:

        * A dict of the current progress
        * An integer count of the number of items to update in this batch.

        The handler should return a deferred integer count of items updated.
        The hander is responsible for updating the progress of the update.

        Args:
            update_name(str): The name of the update that this code handles.
            update_handler(function): The function that does the update.
        """
        self._background_update_handlers[update_name] = update_handler

    def register_background_index_update(self, update_name, index_name,
                                         table, columns, where_clause=None):
        """Helper for store classes to do a background index addition

        To use:

        1. use a schema delta file to add a background update. Example:
            INSERT INTO background_updates (update_name, progress_json) VALUES
                ('my_new_index', '{}');

        2. In the Store constructor, call this method

        Args:
            update_name (str): update_name to register for
            index_name (str): name of index to add
            table (str): table to add index to
            columns (list[str]): columns/expressions to include in index
        """

        # if this is postgres, we add the indexes concurrently. Otherwise
        # we fall back to doing it inline
        if isinstance(self.database_engine, engines.PostgresEngine):
            conc = True
        else:
            conc = False
            # We don't use partial indices on SQLite as it wasn't introduced
            # until 3.8, and wheezy has 3.7
            where_clause = None

        sql = (
            "CREATE INDEX %(conc)s %(name)s ON %(table)s (%(columns)s)"
            " %(where_clause)s"
        ) % {
            "conc": "CONCURRENTLY" if conc else "",
            "name": index_name,
            "table": table,
            "columns": ", ".join(columns),
            "where_clause": "WHERE " + where_clause if where_clause else ""
        }

        def create_index_concurrently(conn):
            conn.rollback()
            # postgres insists on autocommit for the index
            conn.set_session(autocommit=True)
            c = conn.cursor()
            c.execute(sql)
            conn.set_session(autocommit=False)

        def create_index(conn):
            c = conn.cursor()
            c.execute(sql)

        @defer.inlineCallbacks
        def updater(progress, batch_size):
            logger.info("Adding index %s to %s", index_name, table)
            if conc:
                yield self.runWithConnection(create_index_concurrently)
            else:
                yield self.runWithConnection(create_index)
            yield self._end_background_update(update_name)
            defer.returnValue(1)

        self.register_background_update_handler(update_name, updater)

    def start_background_update(self, update_name, progress):
        """Starts a background update running.

        Args:
            update_name: The update to set running.
            progress: The initial state of the progress of the update.

        Returns:
            A deferred that completes once the task has been added to the
            queue.
        """
        # Clear the background update queue so that we will pick up the new
        # task on the next iteration of do_background_update.
        self._background_update_queue = []
        progress_json = json.dumps(progress)

        return self._simple_insert(
            "background_updates",
            {"update_name": update_name, "progress_json": progress_json}
        )

    def _end_background_update(self, update_name):
        """Removes a completed background update task from the queue.

        Args:
            update_name(str): The name of the completed task to remove
        Returns:
            A deferred that completes once the task is removed.
        """
        self._background_update_queue = [
            name for name in self._background_update_queue if name != update_name
        ]
        return self._simple_delete_one(
            "background_updates", keyvalues={"update_name": update_name}
        )

    def _background_update_progress_txn(self, txn, update_name, progress):
        """Update the progress of a background update

        Args:
            txn(cursor): The transaction.
            update_name(str): The name of the background update task
            progress(dict): The progress of the update.
        """

        progress_json = json.dumps(progress)

        self._simple_update_one_txn(
            txn,
            "background_updates",
            keyvalues={"update_name": update_name},
            updatevalues={"progress_json": progress_json},
        )
