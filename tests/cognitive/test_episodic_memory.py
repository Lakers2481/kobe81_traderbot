"""
Unit tests for cognitive/episodic_memory.py

Tests the AI's experiential journal - the complete memory of trading experiences.
"""
import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path


class TestEpisodeOutcomeEnum:
    """Tests for the EpisodeOutcome enumeration."""

    def test_outcome_values(self):
        from cognitive.episodic_memory import EpisodeOutcome

        assert EpisodeOutcome.WIN.value == "win"
        assert EpisodeOutcome.LOSS.value == "loss"
        assert EpisodeOutcome.BREAKEVEN.value == "breakeven"
        assert EpisodeOutcome.STAND_DOWN.value == "stand_down"
        assert EpisodeOutcome.PENDING.value == "pending"


class TestEpisodeDataclass:
    """Tests for the Episode dataclass."""

    def test_episode_creation(self):
        from cognitive.episodic_memory import Episode, EpisodeOutcome

        episode = Episode(
            episode_id="test_123",
            started_at=datetime.now(),
            market_context={'regime': 'BULL'},
            signal_context={'strategy': 'ibs_rsi', 'symbol': 'AAPL'},
        )

        assert episode.episode_id == "test_123"
        assert episode.market_context['regime'] == 'BULL'
        assert episode.outcome == EpisodeOutcome.PENDING
        assert episode.pnl == 0.0

    def test_episode_to_dict(self):
        from cognitive.episodic_memory import Episode

        episode = Episode(
            episode_id="test_456",
            started_at=datetime.now(),
        )
        d = episode.to_dict()

        assert d['episode_id'] == "test_456"
        assert 'started_at' in d
        assert d['outcome'] == 'pending'

    def test_episode_from_dict(self):
        from cognitive.episodic_memory import Episode, EpisodeOutcome

        data = {
            'episode_id': 'test_789',
            'started_at': '2025-01-01T10:00:00',
            'completed_at': '2025-01-01T11:00:00',
            'market_context': {'regime': 'BEAR'},
            'signal_context': {'strategy': 'turtle_soup'},
            'outcome': 'win',
            'pnl': 500.0,
        }
        episode = Episode.from_dict(data)

        assert episode.episode_id == 'test_789'
        assert episode.market_context['regime'] == 'BEAR'
        assert episode.outcome == EpisodeOutcome.WIN
        assert episode.pnl == 500.0

    def test_episode_context_signature(self):
        from cognitive.episodic_memory import Episode

        episode1 = Episode(
            episode_id="ep1",
            started_at=datetime.now(),
            market_context={'regime': 'BULL'},
            signal_context={'strategy': 'ibs_rsi', 'side': 'BUY'},
        )
        episode2 = Episode(
            episode_id="ep2",
            started_at=datetime.now(),
            market_context={'regime': 'BULL'},
            signal_context={'strategy': 'ibs_rsi', 'side': 'BUY'},
        )

        # Same context should produce same signature
        assert episode1.context_signature() == episode2.context_signature()

    def test_episode_different_context_different_signature(self):
        from cognitive.episodic_memory import Episode

        episode1 = Episode(
            episode_id="ep1",
            started_at=datetime.now(),
            market_context={'regime': 'BULL'},
            signal_context={'strategy': 'ibs_rsi'},
        )
        episode2 = Episode(
            episode_id="ep2",
            started_at=datetime.now(),
            market_context={'regime': 'BEAR'},
            signal_context={'strategy': 'turtle_soup'},
        )

        assert episode1.context_signature() != episode2.context_signature()


class TestEpisodicMemoryInitialization:
    """Tests for EpisodicMemory initialization."""

    def test_initialization_creates_directory(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "test_episodes"
            memory = EpisodicMemory(storage_dir=str(storage_path), auto_persist=False)

            assert storage_path.exists()
            assert memory.max_episodes == 1000

    def test_initialization_with_custom_max_episodes(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(
                storage_dir=str(Path(tmpdir) / "episodes"),
                max_episodes=500,
                auto_persist=False,
            )
            assert memory.max_episodes == 500


class TestEpisodeLifecycle:
    """Tests for the full episode lifecycle: start -> add data -> complete."""

    def test_start_episode(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            episode_id = memory.start_episode(
                market_context={'regime': 'BULL', 'vix': 15},
                signal_context={'strategy': 'ibs_rsi', 'symbol': 'AAPL'},
            )

            assert episode_id is not None
            assert len(episode_id) > 10
            assert episode_id in memory._active_episodes

    def test_add_reasoning(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)
            episode_id = memory.start_episode({}, {'strategy': 'test'})

            memory.add_reasoning(episode_id, "RSI is oversold")
            memory.add_reasoning(episode_id, "Market regime is bullish")

            episode = memory._active_episodes[episode_id]
            assert len(episode.reasoning_trace) == 2
            assert "RSI is oversold" in episode.reasoning_trace

    def test_add_reasoning_list(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)
            episode_id = memory.start_episode({}, {'strategy': 'test'})

            memory.add_reasoning(episode_id, ["Step 1", "Step 2", "Step 3"])

            episode = memory._active_episodes[episode_id]
            assert len(episode.reasoning_trace) == 3

    def test_add_concern(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)
            episode_id = memory.start_episode({}, {'strategy': 'test'})

            memory.add_concern(episode_id, "High VIX environment")

            episode = memory._active_episodes[episode_id]
            assert "High VIX environment" in episode.concerns_noted

    def test_add_concerns_multiple(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)
            episode_id = memory.start_episode({}, {'strategy': 'test'})

            memory.add_concerns(episode_id, ["Concern A", "Concern B"])

            episode = memory._active_episodes[episode_id]
            assert len(episode.concerns_noted) == 2

    def test_add_action(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)
            episode_id = memory.start_episode({}, {'strategy': 'test'})

            memory.add_action(episode_id, {'type': 'buy', 'symbol': 'AAPL', 'shares': 100}, decision_mode='fast')

            episode = memory._active_episodes[episode_id]
            assert episode.action_taken['type'] == 'buy'
            assert episode.decision_mode == 'fast'

    def test_complete_episode_win(self):
        from cognitive.episodic_memory import EpisodicMemory, EpisodeOutcome

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)
            episode_id = memory.start_episode(
                {'regime': 'BULL'},
                {'strategy': 'ibs_rsi', 'symbol': 'AAPL'}
            )

            memory.complete_episode(episode_id, {'pnl': 500.0, 'won': True})

            assert episode_id not in memory._active_episodes
            assert episode_id in memory._episodes
            assert memory._episodes[episode_id].outcome == EpisodeOutcome.WIN
            assert memory._episodes[episode_id].pnl == 500.0

    def test_complete_episode_loss(self):
        from cognitive.episodic_memory import EpisodicMemory, EpisodeOutcome

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)
            episode_id = memory.start_episode({}, {'strategy': 'test'})

            memory.complete_episode(episode_id, {'pnl': -200.0, 'won': False})

            assert memory._episodes[episode_id].outcome == EpisodeOutcome.LOSS

    def test_complete_episode_stand_down(self):
        from cognitive.episodic_memory import EpisodicMemory, EpisodeOutcome

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)
            episode_id = memory.start_episode({}, {'strategy': 'test'})

            memory.complete_episode(episode_id, {'stand_down': True})

            assert memory._episodes[episode_id].outcome == EpisodeOutcome.STAND_DOWN


class TestFindingSimilarEpisodes:
    """Tests for finding similar past experiences."""

    def test_find_similar_empty(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            similar = memory.find_similar({'regime': 'BULL', 'strategy': 'ibs_rsi'})
            assert similar == []

    def test_find_similar_with_matching_context(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            # Create matching episodes
            for i in range(3):
                ep_id = memory.start_episode(
                    {'regime': 'BULL'},
                    {'strategy': 'ibs_rsi', 'side': 'BUY'}
                )
                memory.complete_episode(ep_id, {'pnl': 100 * i, 'won': True})

            similar = memory.find_similar({'regime': 'BULL', 'strategy': 'ibs_rsi', 'side': 'BUY'})
            assert len(similar) == 3

    def test_find_similar_respects_limit(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            for i in range(10):
                ep_id = memory.start_episode(
                    {'regime': 'BEAR'},
                    {'strategy': 'turtle_soup', 'side': 'BUY'}
                )
                memory.complete_episode(ep_id, {'pnl': 50, 'won': True})

            similar = memory.find_similar({'regime': 'BEAR', 'strategy': 'turtle_soup', 'side': 'BUY'}, limit=3)
            assert len(similar) == 3


class TestLessonsForContext:
    """Tests for aggregating lessons from past experiences."""

    def test_get_lessons_for_context_empty(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            lessons = memory.get_lessons_for_context({'regime': 'BULL'})
            assert lessons == []

    def test_get_lessons_includes_postmortem_lessons(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            ep_id = memory.start_episode(
                {'regime': 'BULL'},
                {'strategy': 'ibs_rsi', 'side': 'BUY'}
            )
            memory.complete_episode(ep_id, {'pnl': 500, 'won': True})
            memory.add_postmortem(ep_id, "Great trade", lessons=["Wait for confirmation"])

            lessons = memory.get_lessons_for_context({'regime': 'BULL', 'strategy': 'ibs_rsi', 'side': 'BUY'})
            assert "Wait for confirmation" in lessons


class TestWinRateForContext:
    """Tests for calculating historical win rates."""

    def test_win_rate_empty(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            win_rate, sample_size = memory.get_win_rate_for_context({'regime': 'BULL'})
            assert win_rate == 0.0
            assert sample_size == 0

    def test_win_rate_calculation(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            # 6 wins, 4 losses = 60% win rate
            # Note: find_similar() uses a context signature that includes regime, strategy, side
            # Each episode needs a unique signal_context to get a unique ID
            for i in range(10):
                ep_id = memory.start_episode(
                    {'regime': 'NEUTRAL'},
                    {'strategy': 'test', 'side': 'BUY', 'unique_id': i}  # Add unique identifier
                )
                memory.complete_episode(ep_id, {'pnl': 100 if i < 6 else -50, 'won': i < 6})

            # Verify episodes are in memory
            assert len(memory._episodes) == 10, f"Expected 10 episodes, got {len(memory._episodes)}"

            # Query with the same context that was used to create episodes
            # Note: The 'unique_id' field doesn't affect the context signature (only regime, strategy, side)
            win_rate, sample_size = memory.get_win_rate_for_context({
                'regime': 'NEUTRAL',
                'strategy': 'test',
                'side': 'BUY'
            })

            # Note: Due to context signature matching, all 10 episodes should match
            # But find_similar has a default limit of 100, so we should find all 10
            assert sample_size == 10, f"Expected 10 samples, got {sample_size}"
            assert win_rate == 0.6, f"Expected 0.6 win rate, got {win_rate}"


class TestRecentEpisodes:
    """Tests for retrieving recent episodes."""

    def test_get_recent_episodes(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            for i in range(5):
                ep_id = memory.start_episode({}, {'strategy': f'test_{i}'})
                memory.complete_episode(ep_id, {'pnl': i * 10})

            recent = memory.get_recent_episodes(limit=3)
            assert len(recent) == 3


class TestPostmortem:
    """Tests for adding postmortem reflections to episodes."""

    def test_add_postmortem(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            ep_id = memory.start_episode({}, {'strategy': 'test'})
            memory.complete_episode(ep_id, {'pnl': 300})

            memory.add_postmortem(
                ep_id,
                postmortem="Solid execution",
                lessons=["Patience pays off"],
                what_to_repeat=["Wait for pullback entry"],
            )

            episode = memory.get_episode(ep_id)
            assert episode.postmortem == "Solid execution"
            assert "Patience pays off" in episode.lessons_learned
            assert "Wait for pullback entry" in episode.what_to_repeat


class TestStatistics:
    """Tests for memory statistics."""

    def test_get_stats_empty(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            stats = memory.get_stats()
            assert stats['total_episodes'] == 0

    def test_get_stats_with_episodes(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(storage_dir=str(Path(tmpdir) / "episodes"), auto_persist=False)

            # Create episodes with different context signatures to avoid pruning
            for i in range(5):
                ep_id = memory.start_episode(
                    {'regime': f'REGIME_{i}'},  # Different regime for each
                    {'strategy': f'test_{i}'}
                )
                memory.complete_episode(ep_id, {'pnl': 100 if i < 3 else -50, 'won': i < 3})

            stats = memory.get_stats()
            assert stats['total_episodes'] == 5
            assert stats['wins'] == 3
            assert stats['losses'] == 2


class TestPruning:
    """Tests for episode pruning when memory is full."""

    def test_prune_least_important(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = EpisodicMemory(
                storage_dir=str(Path(tmpdir) / "episodes"),
                max_episodes=5,
                auto_persist=False,
            )

            # Add 7 episodes to a memory with max 5
            for i in range(7):
                ep_id = memory.start_episode({}, {'strategy': f'test_{i}'})
                memory.complete_episode(ep_id, {'pnl': 10})

            assert len(memory._episodes) == 5


class TestPersistence:
    """Tests for saving and loading episodes from disk."""

    def test_save_and_load_episode(self):
        from cognitive.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = str(Path(tmpdir) / "episodes")

            # Create and complete an episode
            memory1 = EpisodicMemory(storage_dir=storage_path, auto_persist=True)
            ep_id = memory1.start_episode({'regime': 'BULL'}, {'strategy': 'ibs_rsi'})
            memory1.complete_episode(ep_id, {'pnl': 250, 'won': True})

            # Create a new memory instance that loads from disk
            memory2 = EpisodicMemory(storage_dir=storage_path, auto_persist=False)

            assert ep_id in memory2._episodes
            assert memory2._episodes[ep_id].pnl == 250


class TestSingletonFactory:
    """Tests for the singleton factory function."""

    def test_get_episodic_memory_singleton(self):
        from cognitive.episodic_memory import get_episodic_memory

        memory1 = get_episodic_memory()
        memory2 = get_episodic_memory()

        assert memory1 is memory2
