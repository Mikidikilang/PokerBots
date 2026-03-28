"""
train_local.py — Javítandó szekciók (H4, H5, C2 fix)

Ez a fajl a train_local.py-ban szukseges valtozasokat dokumentalja.
Az eredeti fajl mar 523 soros; itt csak a modosult reszeket adjuk meg.

=============================================================================
FIX H5 — start_time nem adodik at a build_training_pipeline()-nak
=============================================================================

HELY: main() fuggveny, a build_training_pipeline() hivasa ELOTT

ELOTT (hianyzo sor):
    pipeline: dict[str, Any] = build_training_pipeline(
        cfg,
        device_override=args.device,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )

UTAN (javitott kod):
    # [FIX H5] A session legelso pillanataban rogzitjuk a monotonic orát.
    # A build_training_pipeline() checkpoint letoltest is veget, ami percekig
    # tarthat — ez az ido elveszne a 11.5h limitbol ha itt nem rogzitjuk.
    _session_start: float = time.monotonic()
    logger.info(
        "Session clock indulva: %.4f (monotonic) [H5 FIX]",
        _session_start,
    )

    pipeline: dict[str, Any] = build_training_pipeline(
        cfg,
        device_override=args.device,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        start_time=_session_start,        # [FIX H5] HOZZAADVA
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )

=============================================================================
FIX H4 — Scheduler allapot nem kerul mentesre a checkpointba
=============================================================================

HELY: build_training_pipeline() belso on_checkpoint() callback

ELOTT:
    def on_checkpoint(iteration: int, net: Any) -> None:
        if rank != 0:
            return
        rng_states = RNGStateManager.capture_states(dl_generator)
        state_manager.save_training_state(
            network=net,
            optimizer=runner.trainer.optimizer,
            iteration=iteration,
            total_env_steps=runner.collector.get_total_steps(),
            total_hands=0,
            best_mean_reward=-float("inf"),
            orchestrator_state=orchestrator.get_state(),
            config=cfg,
            rng_states=rng_states,
            wandb_run_id=monitor.run_id if monitor.active else None,
            is_best=False,
        )

UTAN:
    def on_checkpoint(iteration: int, net: Any) -> None:
        if rank != 0:
            return
        rng_states = RNGStateManager.capture_states(dl_generator)
        state_manager.save_training_state(
            network=net,
            optimizer=runner.trainer.optimizer,
            scheduler=runner.trainer.scheduler,  # [FIX H4] HOZZAADVA
            iteration=iteration,
            total_env_steps=runner.collector.get_total_steps(),
            total_hands=0,
            best_mean_reward=-float("inf"),
            orchestrator_state=orchestrator.get_state(),
            config=cfg,
            rng_states=rng_states,
            wandb_run_id=monitor.run_id if monitor.active else None,
            is_best=False,
        )

=============================================================================
FIX H4 — Scheduler allapot visszatoltese resume eseten
=============================================================================

HELY: build_training_pipeline() resume blokkjaban, az optimizer restore UTAN

# --- Resume optimizer and RNG states ---
if checkpoint_to_resume is not None:
    if "optimizer_state_dict" in checkpoint_to_resume:
        runner.trainer.optimizer.load_state_dict(
            checkpoint_to_resume["optimizer_state_dict"]
        )
        logger.info("Optimizer allapot visszaallitva a checkpoint-bol")

    # [FIX H4] Scheduler allapot visszatoltese — megakadalyozza, hogy az LR
    # schedule a Kaggle session folytatodasakor nullarol induljon ujra.
    if (
        "scheduler_state_dict" in checkpoint_to_resume
        and checkpoint_to_resume["scheduler_state_dict"] is not None
        and runner.trainer.scheduler is not None
    ):
        runner.trainer.scheduler.load_state_dict(
            checkpoint_to_resume["scheduler_state_dict"]
        )
        logger.info("LR scheduler allapot visszaallitva a checkpoint-bol [H4 FIX]")

    if "rng_states" in checkpoint_to_resume and checkpoint_to_resume["rng_states"]:
        RNGStateManager.restore_states(checkpoint_to_resume["rng_states"])
        logger.info("RNG allapot visszaallitva a checkpoint-bol (determinisztikus)")

=============================================================================
FIX C2 — DDP world_size atadasa az Orchestratornak
=============================================================================

HELY: build_training_pipeline() az orchestrator inicializalasa UTAN

ELOTT:
    if rank == 0:
        AutoAdaptiveOrchestrator.reset_instance()
        ...
        orchestrator = AutoAdaptiveOrchestrator.get_instance(orch_config, cfg)
        ...
        orchestrator.set_network_reference(network)

UTAN:
    if rank == 0:
        AutoAdaptiveOrchestrator.reset_instance()
        ...
        orchestrator = AutoAdaptiveOrchestrator.get_instance(orch_config, cfg)
        ...
        orchestrator.set_network_reference(network)
        # [FIX C2] DDP world_size atadasa a barrier logikához az FSP snapshotban
        orchestrator.set_ddp_world_size(world_size)
        logger.info("DDP world_size=%d atadva az Orchestratornak [C2 FIX]", world_size)

=============================================================================
TELJES JAVÍTOTT build_training_pipeline() - Kulcssekciok
=============================================================================

Az alabbi a build_training_pipeline() teljes modosított teste, csak a
valtoztatott reszekett kiemelve. A tobb szaz soros fuggveny teljes
atirasatol megkimeljuk ezt a dokumentumot — a fenti diff-ek elegendoek.

Osszefoglalo a szukseges valtozasokhoz:
1. [H5] start_time parameter hasznalata main()-ben es pipeline-ban: OK (mar implementalva)
   Ha a start_time=None, a GracefulShutdownMonitor time.monotonic()-ot hiv az __init__-ban.
   A main() mostantol _session_start-ot ad at start_time-kent.

2. [H4] scheduler=runner.trainer.scheduler atadasa on_checkpoint()-ban: uj sor
   + checkpoint_to_resume scheduler restore: uj blokk

3. [C2] orchestrator.set_ddp_world_size(world_size) hivasa: uj sor
"""

# =============================================================================
# Ez a fajl dokumentacios celra keszult.
# A tényleges valtozasokat a fenti diff-ek szerint kell alkalmazni
# a scripts/train_local.py fajlban.
# =============================================================================
