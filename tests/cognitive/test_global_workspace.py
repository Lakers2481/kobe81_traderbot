"""
Unit tests for cognitive/global_workspace.py

Tests the AI's inter-module communication hub (inspired by Global Workspace Theory).
"""
import pytest
import time


class TestPriorityEnum:
    """Tests for the Priority enumeration."""

    def test_priority_values(self):
        from cognitive.global_workspace import Priority

        assert Priority.CRITICAL.value == 1
        assert Priority.HIGH.value == 2
        assert Priority.NORMAL.value == 3
        assert Priority.LOW.value == 4
        assert Priority.BACKGROUND.value == 5


class TestWorkspaceItemDataclass:
    """Tests for the WorkspaceItem dataclass."""

    def test_item_creation(self):
        from cognitive.global_workspace import WorkspaceItem, Priority

        item = WorkspaceItem(
            topic="regime",
            data={'state': 'BULL', 'confidence': 0.85},
            priority=Priority.NORMAL,
            source="test_module",
        )

        assert item.topic == "regime"
        assert item.data['state'] == 'BULL'
        assert item.priority == Priority.NORMAL
        assert item.source == "test_module"

    def test_item_with_custom_ttl(self):
        from cognitive.global_workspace import WorkspaceItem, Priority

        item = WorkspaceItem(
            topic="short_lived",
            data={'value': 42},
            priority=Priority.HIGH,
            source="test",
            ttl_seconds=60,
        )

        assert item.ttl_seconds == 60

    def test_item_to_dict(self):
        from cognitive.global_workspace import WorkspaceItem, Priority

        item = WorkspaceItem(
            topic="test_topic",
            data={'key': 'value'},
            priority=Priority.NORMAL,
            source="test",
        )
        d = item.to_dict()

        assert d['topic'] == "test_topic"
        assert d['source'] == "test"
        assert d['priority'] == "NORMAL"

    def test_item_is_expired(self):
        from cognitive.global_workspace import WorkspaceItem, Priority
        from datetime import datetime, timedelta

        # Create an item that's already expired
        item = WorkspaceItem(
            topic="expired",
            data={},
            priority=Priority.LOW,
            source="test",
            ttl_seconds=0,
        )
        # Manually set timestamp to the past
        item.timestamp = datetime.now() - timedelta(seconds=1)

        assert item.is_expired() is True


class TestGlobalWorkspaceInitialization:
    """Tests for GlobalWorkspace initialization."""

    def test_default_initialization(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        assert workspace is not None
        assert workspace.working_memory_capacity == 7

    def test_custom_capacity(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace(working_memory_capacity=5)

        assert workspace.working_memory_capacity == 5


class TestPublishSubscribe:
    """Tests for the publish/subscribe mechanism."""

    def test_subscribe_to_topic(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        received_items = []

        def callback(item):
            received_items.append(item)

        workspace.subscribe("test_topic", callback)

        assert "test_topic" in workspace._subscribers
        assert len(workspace._subscribers["test_topic"]) == 1

    def test_publish_message(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        received_items = []

        def callback(item):
            received_items.append(item)

        workspace.subscribe("alerts", callback)
        workspace.publish(topic="alerts", data={'alert': 'Test alert'}, source="test_module")

        assert len(received_items) == 1
        assert received_items[0].data['alert'] == 'Test alert'

    def test_multiple_subscribers(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        received_1 = []
        received_2 = []

        workspace.subscribe("shared_topic", lambda item: received_1.append(item))
        workspace.subscribe("shared_topic", lambda item: received_2.append(item))

        workspace.publish(topic="shared_topic", data={'data': 'shared'}, source="sender")

        assert len(received_1) == 1
        assert len(received_2) == 1

    def test_publish_to_nonexistent_topic(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        # Should not raise an error
        workspace.publish(topic="nonexistent_topic", data={'data': 'test'}, source="sender")

    def test_wildcard_subscription(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()
        received = []

        workspace.subscribe("*", lambda item: received.append(item))
        workspace.publish(topic="any_topic", data={'test': 'data'}, source="sender")

        assert len(received) == 1


class TestGetData:
    """Tests for retrieving data from the workspace."""

    def test_get_published_data(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        workspace.publish(topic="current_regime", data="BULL", source="regime_detector")
        value = workspace.get("current_regime")

        assert value == "BULL"

    def test_get_nonexistent_data(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        value = workspace.get("nonexistent_key")
        assert value is None

    def test_get_with_default(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        value = workspace.get("missing_key", default="default_value")
        assert value == "default_value"

    def test_get_item_full(self):
        from cognitive.global_workspace import GlobalWorkspace, WorkspaceItem

        workspace = GlobalWorkspace()

        workspace.publish(topic="regime", data={'state': 'BULL'}, source="test")
        item = workspace.get_item("regime")

        assert isinstance(item, WorkspaceItem)
        assert item.data['state'] == 'BULL'


class TestGetContext:
    """Tests for getting the full context snapshot."""

    def test_get_context_multiple_topics(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        workspace.publish(topic="regime", data="BULL", source="test")
        workspace.publish(topic="vix", data=15.5, source="test")
        workspace.publish(topic="signal", data={'symbol': 'AAPL'}, source="test")

        context = workspace.get_context()

        assert context['regime'] == "BULL"
        assert context['vix'] == 15.5
        assert context['signal']['symbol'] == 'AAPL'

    def test_get_context_empty(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        context = workspace.get_context()
        assert context == {}


class TestWorkingMemory:
    """Tests for the working memory functionality."""

    def test_high_priority_enters_working_memory(self):
        from cognitive.global_workspace import GlobalWorkspace, Priority

        workspace = GlobalWorkspace()

        workspace.publish(
            topic="critical_alert",
            data={'alert': 'KILL_SWITCH'},
            priority=Priority.CRITICAL,
            source="risk_gate",
        )

        wm = workspace.get_working_memory()
        topics = [item['topic'] for item in wm]

        assert "critical_alert" in topics

    def test_low_priority_does_not_enter_working_memory(self):
        from cognitive.global_workspace import GlobalWorkspace, Priority

        workspace = GlobalWorkspace()

        workspace.publish(
            topic="background_info",
            data={'info': 'status update'},
            priority=Priority.BACKGROUND,
            source="health_check",
        )

        wm = workspace.get_working_memory()
        topics = [item['topic'] for item in wm]

        assert "background_info" not in topics

    def test_focus_attention(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        workspace.publish(topic="target", data={'value': 42}, source="test")
        data = workspace.focus_attention("target")

        assert data == {'value': 42}

    def test_working_memory_capacity_limit(self):
        from cognitive.global_workspace import GlobalWorkspace, Priority

        workspace = GlobalWorkspace(working_memory_capacity=3)

        # Publish more than capacity
        for i in range(5):
            workspace.publish(
                topic=f"item_{i}",
                data={'index': i},
                priority=Priority.HIGH,
                source="test",
            )

        wm = workspace.get_working_memory()

        assert len(wm) <= 3


class TestStatistics:
    """Tests for workspace statistics."""

    def test_get_stats(self):
        from cognitive.global_workspace import GlobalWorkspace

        workspace = GlobalWorkspace()

        workspace.subscribe("topic1", lambda item: None)
        workspace.subscribe("topic2", lambda item: None)
        workspace.publish(topic="topic1", data={'msg': 'test'}, source="sender")

        stats = workspace.get_stats()

        assert 'active_topics' in stats
        assert 'total_publishes' in stats
        assert stats['total_publishes'] >= 1


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_expired_items(self):
        from cognitive.global_workspace import GlobalWorkspace, WorkspaceItem, Priority
        from datetime import datetime, timedelta

        workspace = GlobalWorkspace()

        # Manually add an expired item
        expired_item = WorkspaceItem(
            topic="expired_topic",
            data={},
            priority=Priority.LOW,
            source="test",
            ttl_seconds=0,
        )
        expired_item.timestamp = datetime.now() - timedelta(seconds=10)
        workspace._workspace["expired_topic"] = expired_item

        expired_count = workspace.cleanup()

        assert expired_count >= 1
        assert "expired_topic" not in workspace._workspace


class TestDumpState:
    """Tests for state dumping."""

    def test_dump_state_json(self):
        from cognitive.global_workspace import GlobalWorkspace
        import json

        workspace = GlobalWorkspace()

        workspace.publish(topic="test", data="value", source="test")
        state_json = workspace.dump_state()

        # Should be valid JSON
        state = json.loads(state_json)
        assert 'workspace' in state
        assert 'stats' in state


class TestSingletonFactory:
    """Tests for the singleton factory function."""

    def test_get_workspace(self):
        from cognitive.global_workspace import get_workspace

        workspace1 = get_workspace()
        workspace2 = get_workspace()

        assert workspace1 is workspace2


class TestTopicConstants:
    """Tests for predefined topic constants."""

    def test_topic_constants_exist(self):
        from cognitive.global_workspace import GlobalWorkspace

        assert hasattr(GlobalWorkspace, 'TOPIC_REGIME')
        assert hasattr(GlobalWorkspace, 'TOPIC_SIGNAL')
        assert hasattr(GlobalWorkspace, 'TOPIC_RISK')
        assert hasattr(GlobalWorkspace, 'TOPIC_CONFIDENCE')
        assert hasattr(GlobalWorkspace, 'TOPIC_INSIGHT')
        assert hasattr(GlobalWorkspace, 'TOPIC_ERROR')
        assert hasattr(GlobalWorkspace, 'TOPIC_ALERT')
