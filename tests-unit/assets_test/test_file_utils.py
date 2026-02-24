from app.assets.services.file_utils import is_visible, list_files_recursively


class TestIsVisible:
    def test_visible_file(self):
        assert is_visible("file.txt") is True

    def test_hidden_file(self):
        assert is_visible(".hidden") is False

    def test_hidden_directory(self):
        assert is_visible(".git") is False

    def test_visible_directory(self):
        assert is_visible("src") is True

    def test_dotdot_is_hidden(self):
        assert is_visible("..") is False

    def test_dot_is_hidden(self):
        assert is_visible(".") is False


class TestListFilesRecursively:
    def test_skips_hidden_files(self, tmp_path):
        (tmp_path / "visible.txt").write_text("a")
        (tmp_path / ".hidden").write_text("b")

        result = list_files_recursively(str(tmp_path))

        assert len(result) == 1
        assert result[0].endswith("visible.txt")

    def test_skips_hidden_directories(self, tmp_path):
        hidden_dir = tmp_path / ".hidden_dir"
        hidden_dir.mkdir()
        (hidden_dir / "file.txt").write_text("a")

        visible_dir = tmp_path / "visible_dir"
        visible_dir.mkdir()
        (visible_dir / "file.txt").write_text("b")

        result = list_files_recursively(str(tmp_path))

        assert len(result) == 1
        assert "visible_dir" in result[0]
        assert ".hidden_dir" not in result[0]

    def test_empty_directory(self, tmp_path):
        result = list_files_recursively(str(tmp_path))
        assert result == []

    def test_nonexistent_directory(self, tmp_path):
        result = list_files_recursively(str(tmp_path / "nonexistent"))
        assert result == []
