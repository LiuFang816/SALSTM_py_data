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


class MediaRepositoryStore(SQLBaseStore):
    """Persistence for attachments and avatars"""

    def get_default_thumbnails(self, top_level_type, sub_type):
        return []

    def get_local_media(self, media_id):
        """Get the metadata for a local piece of media
        Returns:
            None if the media_id doesn't exist.
        """
        return self._simple_select_one(
            "local_media_repository",
            {"media_id": media_id},
            ("media_type", "media_length", "upload_name", "created_ts"),
            allow_none=True,
            desc="get_local_media",
        )

    def store_local_media(self, media_id, media_type, time_now_ms, upload_name,
                          media_length, user_id):
        return self._simple_insert(
            "local_media_repository",
            {
                "media_id": media_id,
                "media_type": media_type,
                "created_ts": time_now_ms,
                "upload_name": upload_name,
                "media_length": media_length,
                "user_id": user_id.to_string(),
            },
            desc="store_local_media",
        )

    def get_url_cache(self, url, ts):
        """Get the media_id and ts for a cached URL as of the given timestamp
        Returns:
            None if the URL isn't cached.
        """
        def get_url_cache_txn(txn):
            # get the most recently cached result (relative to the given ts)
            sql = (
                "SELECT response_code, etag, expires, og, media_id, download_ts"
                " FROM local_media_repository_url_cache"
                " WHERE url = ? AND download_ts <= ?"
                " ORDER BY download_ts DESC LIMIT 1"
            )
            txn.execute(sql, (url, ts))
            row = txn.fetchone()

            if not row:
                # ...or if we've requested a timestamp older than the oldest
                # copy in the cache, return the oldest copy (if any)
                sql = (
                    "SELECT response_code, etag, expires, og, media_id, download_ts"
                    " FROM local_media_repository_url_cache"
                    " WHERE url = ? AND download_ts > ?"
                    " ORDER BY download_ts ASC LIMIT 1"
                )
                txn.execute(sql, (url, ts))
                row = txn.fetchone()

            if not row:
                return None

            return dict(zip((
                'response_code', 'etag', 'expires', 'og', 'media_id', 'download_ts'
            ), row))

        return self.runInteraction(
            "get_url_cache", get_url_cache_txn
        )

    def store_url_cache(self, url, response_code, etag, expires, og, media_id,
                        download_ts):
        return self._simple_insert(
            "local_media_repository_url_cache",
            {
                "url": url,
                "response_code": response_code,
                "etag": etag,
                "expires": expires,
                "og": og,
                "media_id": media_id,
                "download_ts": download_ts,
            },
            desc="store_url_cache",
        )

    def get_local_media_thumbnails(self, media_id):
        return self._simple_select_list(
            "local_media_repository_thumbnails",
            {"media_id": media_id},
            (
                "thumbnail_width", "thumbnail_height", "thumbnail_method",
                "thumbnail_type", "thumbnail_length",
            ),
            desc="get_local_media_thumbnails",
        )

    def store_local_thumbnail(self, media_id, thumbnail_width,
                              thumbnail_height, thumbnail_type,
                              thumbnail_method, thumbnail_length):
        return self._simple_insert(
            "local_media_repository_thumbnails",
            {
                "media_id": media_id,
                "thumbnail_width": thumbnail_width,
                "thumbnail_height": thumbnail_height,
                "thumbnail_method": thumbnail_method,
                "thumbnail_type": thumbnail_type,
                "thumbnail_length": thumbnail_length,
            },
            desc="store_local_thumbnail",
        )

    def get_cached_remote_media(self, origin, media_id):
        return self._simple_select_one(
            "remote_media_cache",
            {"media_origin": origin, "media_id": media_id},
            (
                "media_type", "media_length", "upload_name", "created_ts",
                "filesystem_id",
            ),
            allow_none=True,
            desc="get_cached_remote_media",
        )

    def store_cached_remote_media(self, origin, media_id, media_type,
                                  media_length, time_now_ms, upload_name,
                                  filesystem_id):
        return self._simple_insert(
            "remote_media_cache",
            {
                "media_origin": origin,
                "media_id": media_id,
                "media_type": media_type,
                "media_length": media_length,
                "created_ts": time_now_ms,
                "upload_name": upload_name,
                "filesystem_id": filesystem_id,
                "last_access_ts": time_now_ms,
            },
            desc="store_cached_remote_media",
        )

    def update_cached_last_access_time(self, origin_id_tuples, time_ts):
        def update_cache_txn(txn):
            sql = (
                "UPDATE remote_media_cache SET last_access_ts = ?"
                " WHERE media_origin = ? AND media_id = ?"
            )

            txn.executemany(sql, (
                (time_ts, media_origin, media_id)
                for media_origin, media_id in origin_id_tuples
            ))

        return self.runInteraction("update_cached_last_access_time", update_cache_txn)

    def get_remote_media_thumbnails(self, origin, media_id):
        return self._simple_select_list(
            "remote_media_cache_thumbnails",
            {"media_origin": origin, "media_id": media_id},
            (
                "thumbnail_width", "thumbnail_height", "thumbnail_method",
                "thumbnail_type", "thumbnail_length", "filesystem_id",
            ),
            desc="get_remote_media_thumbnails",
        )

    def store_remote_media_thumbnail(self, origin, media_id, filesystem_id,
                                     thumbnail_width, thumbnail_height,
                                     thumbnail_type, thumbnail_method,
                                     thumbnail_length):
        return self._simple_insert(
            "remote_media_cache_thumbnails",
            {
                "media_origin": origin,
                "media_id": media_id,
                "thumbnail_width": thumbnail_width,
                "thumbnail_height": thumbnail_height,
                "thumbnail_method": thumbnail_method,
                "thumbnail_type": thumbnail_type,
                "thumbnail_length": thumbnail_length,
                "filesystem_id": filesystem_id,
            },
            desc="store_remote_media_thumbnail",
        )

    def get_remote_media_before(self, before_ts):
        sql = (
            "SELECT media_origin, media_id, filesystem_id"
            " FROM remote_media_cache"
            " WHERE last_access_ts < ?"
        )

        return self._execute(
            "get_remote_media_before", self.cursor_to_dict, sql, before_ts
        )

    def delete_remote_media(self, media_origin, media_id):
        def delete_remote_media_txn(txn):
            self._simple_delete_txn(
                txn,
                "remote_media_cache",
                keyvalues={
                    "media_origin": media_origin, "media_id": media_id
                },
            )
            self._simple_delete_txn(
                txn,
                "remote_media_cache_thumbnails",
                keyvalues={
                    "media_origin": media_origin, "media_id": media_id
                },
            )
        return self.runInteraction("delete_remote_media", delete_remote_media_txn)
