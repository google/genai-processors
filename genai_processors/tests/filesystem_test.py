# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import shutil
from typing import List

from absl.testing import absltest
from genai_processors import processor
from genai_processors.core import filesystem


class FilesystemTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.test_dir = os.path.join(
        absltest.get_default_test_tmpdir(), 'filesystem_test'
    )
    os.makedirs(self.test_dir, exist_ok=True)

  def tearDown(self) -> None:
    super().tearDown()
    shutil.rmtree(self.test_dir)

  def create_file(self, filename: str, content: str) -> None:
    """Creates a file with the given content in the test directory."""
    filepath = os.path.join(self.test_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
      f.write(content)

  def create_files(self, filenames: List[str]) -> None:
    """Creates the given files."""
    for name in filenames:
      self.create_file(name, 'foo')

  def test_glob_source(self):
    self.create_files(
        ['todo.txt', 'cat.jpeg', 'first_steps.mp4', 'junk/important.txt']
    )

    # Test with a pattern that matches multiple files
    p = filesystem.GlobSource(pattern='**/*.txt', base_dir=self.test_dir)
    results = processor.apply_sync(p, [])
    # Part order is preserved.
    filenames = [part.metadata['original_file_name'] for part in results]
    self.assertEqual(filenames, ['junk/important.txt', 'todo.txt'])
    for r in results:
      self.assertEqual(r.mimetype, 'text/plain')

    # Test with a pattern that matches one file
    p = filesystem.GlobSource(pattern='*.jpeg', base_dir=self.test_dir)
    results = processor.apply_sync(p, [])
    self.assertLen(results, 1)
    filenames = [part.metadata['original_file_name'] for part in results]
    self.assertEqual(filenames[0], 'cat.jpeg')
    self.assertEqual(results[0].mimetype, 'image/jpeg')

    # Test with a pattern that matches no files
    p = filesystem.GlobSource(pattern='*.zip', base_dir=self.test_dir)
    results = processor.apply_sync(p, [])
    self.assertEmpty(results)

  def test_glob_source_no_inline(self):
    self.create_files(['foo.txt'])
    p = filesystem.GlobSource(
        pattern='**/*.txt', base_dir=self.test_dir, inline_file_data=False
    )
    results = processor.apply_sync(p, [])
    self.assertLen(results, 1)
    self.assertIsNotNone(results[0].part.file_data)
    file_data = results[0].part.file_data
    self.assertEqual(file_data.display_name, 'foo.txt')
    self.assertEqual(file_data.file_uri, os.path.join(self.test_dir, 'foo.txt'))

  def test_glob_source_natural_sort(self):
    # Create some test files with names that require natural sorting
    self.create_files(['f2.txt', 'f10.txt', 'f1.txt'])

    # Test if GlobSource returns them in natural order
    p = filesystem.GlobSource(pattern='*.txt', base_dir=self.test_dir)
    results = processor.apply_sync(p, [])
    filenames = [part.metadata['original_file_name'] for part in results]
    self.assertEqual(
        filenames,
        [
            'f1.txt',
            'f2.txt',
            'f10.txt',
        ],
    )


if __name__ == '__main__':
  absltest.main()
