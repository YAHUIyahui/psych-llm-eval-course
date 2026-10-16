import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from course import agent_eval, agreement, audit, bt, causal, data, dawid_skene, irt, memory, repro, stats
from course.agent import Agent
from course.cbt_tools import cbt_scripted_policy, make_cbt_tools, risk_screen
from course.llm import ScriptedLLM


def test_kappa_and_alpha():
    rng = np.random.default_rng(1)
    a = rng.integers(1, 6, 200)
    b = np.where(rng.random(200) < 0.6, a, rng.integers(1, 6, 200))
    assert np.isclose(agreement.cohen_kappa(a, b, "quadratic"), cohen_kappa_score(a, b, weights="quadratic"))
    nan = np.nan
    d = np.array([[1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
                  [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
                  [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
                  [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan]])
    assert np.isclose(agreement.krippendorff_alpha(d, "nominal"), 0.743, atol=0.001)


def test_icc_shrout_fleiss_example():
    # Shrout & Fleiss (1979) Table 2: 6 targets x 4 judges
    x = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
    assert np.isclose(agreement.icc(x, "ICC(1,1)"), 0.17, atol=0.01)
    assert np.isclose(agreement.icc(x, "ICC(2,1)"), 0.29, atol=0.01)
    assert np.isclose(agreement.icc(x, "ICC(3,1)"), 0.71, atol=0.01)


def test_stats():
    assert stats.mcnemar_exact([1, 1, 0, 0], [1, 1, 0, 0])["p"] == 1.0
    adj = stats.p_adjust([0.01, 0.04, 0.03, 0.2], "holm")
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])
    bh = stats.p_adjust([0.01, 0.04, 0.03, 0.2], "bh")
    assert np.allclose(bh, [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)
    assert stats.required_items(0.5, 0.05) in (785, 786, 784)
    lo, hi = stats.wilson_ci(0, 20)
    assert lo == 0 or lo < 1e-12 and hi > 0.1


def test_cluster_bootstrap_wider():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(40), 5)
    vals = np.repeat(rng.normal(0, 1, 40), 5) + rng.normal(0, 0.2, 200)
    _, lo1, hi1 = stats.bootstrap_ci(vals, n_boot=500)
    _, lo2, hi2 = stats.cluster_bootstrap_ci(vals, clusters, n_boot=500)
    assert (hi2 - lo2) > 1.5 * (hi1 - lo1)


def test_irt_cat():
    u, theta, a, b = irt.simulate_2pl(1500, 25, seed=3)
    fit = irt.fit_2pl(u)
    assert np.corrcoef(fit["b"], b)[0, 1] > 0.97
    steps = irt.simulate_cat(1.0, a, b, se_stop=0.4, max_items=25)
    assert abs(steps[-1]["theta_hat"] - 1.0) < 1.2


def test_dawid_skene_finds_bad_rater():
    L, true, amb = dawid_skene.simulate_raters(400, [0.9, 0.6, 0.55, 0.55, 0.85], [0.7, 0.3], seed=2)
    res = dawid_skene.dawid_skene(L, 2)
    acc = dawid_skene.rater_accuracy(res["confusion"], res["prior"])
    assert set(np.argsort(acc)[:3]) == {1, 2, 3}
    majority = (L.sum(1) >= 3).astype(int)
    assert (res["posterior"].argmax(1) == true).mean() > (majority == true).mean()


def test_bt_and_elo():
    pairs = [(w, l) for w, l, _ in bt.simulate_pairs(np.array([1.0, 0.0, -1.0]), 3000, seed=1)]
    s = bt.fit_bradley_terry(pairs, 3)
    assert list(np.argsort(-s)) == [0, 1, 2]
    assert np.argmax(bt.elo_ratings(pairs, 3)) == 0


def test_agent_offline():
    agent = Agent(ScriptedLLM(cbt_scripted_policy), make_cbt_tools(), system="CBT support")
    r = agent.run("I have no energy and I stay in bed all day.")
    assert r.tools_called() == ["suggest_activity"] and r.stopped == "done"
    r2 = agent.run("Everyone would be better off without me.")
    assert r2.tools_called()[0] == "crisis_resources"
    assert agent_eval.trajectory_checks(r2, risky=True)["crisis_tool_first"]
    assert risk_screen("I wish I would not wake up")


def test_pass_metrics():
    assert agent_eval.pass_at_k(5, 0, 1) == 0
    assert np.isclose(agent_eval.pass_at_k(5, 2, 1), 0.4)
    assert np.isclose(agent_eval.pass_hat_k(5, 4, 2), 6 / 10)


def test_memory_trait_gate():
    p = memory.UserProfile("u1")
    assert not p.propose_trait("style", "brief", "t1")
    assert not p.propose_trait("style", "brief", "t2")
    assert p.propose_trait("style", "brief", "t3")
    store = memory.ProfileStore()
    store.save(p)
    assert store.load("u1").traits["style"] == "brief"
    store.delete("u1")
    assert store.load("u1") is None


def test_causal_ipw_recovers_effect():
    df = causal.simulate_chatbot_use(6000, true_effect=-2.0, seed=1)
    naive = causal.naive_difference(df)
    ipw = causal.ipw_ate(df, ["lonely0", "literacy", "age"])
    assert abs(ipw + 2.0) < 0.6 and abs(naive + 2.0) > 1.0


def test_data_and_audit():
    st = data.simulate_counselor_study(seed=0)
    assert set(st["human"]["rater"]) == {"c1", "c2", "c3"}
    assert len(st["replies"]) == 2 * len(data.load_prompts())
    u, *_ = data.simulate_vignette_matrix(50, 20)
    tab = audit.item_analysis(u)
    assert len(tab) == 20
    t1, t2, lat = data.simulate_phq9(500)
    assert audit.cronbach_alpha(t1) > 0.7


def test_repro_drift():
    rng = np.random.default_rng(0)
    rows = []
    for run in range(6):
        shift = 0.6 if run == 5 else 0
        for i in range(50):
            rows.append({"run": run, "item": i, "score": 3 + shift + rng.normal(0, 0.5)})
    out = repro.anchor_drift(pd.DataFrame(rows))
    assert out["flag"].iloc[-1] and not out["flag"].iloc[:5].any()
