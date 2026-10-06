# Sample meeting script (for the shareable demo recording)

Record this with 2–3 teammates (a phone voice recorder is fine, ~5 minutes). Read naturally, and
ad-lib a little. The script is designed to test every point the judges grade:

| Test | Where |
|---|---|
| Domain terms likely to be misheard (Kubernetes, PyTorch, ONNX, Kafka, Grafana, CI/CD, p95) | throughout |
| Task with owner **and** deadline | Priya → dashboard by Thursday |
| Task with owner, **no** deadline | Arjun → Kafka consumer lag |
| Task with **no** owner | load-test script |
| Proposal that is **not** agreed (must not become a decision) | switching to TensorRT |
| Item explicitly **deferred** | GPU budget |
| **Negative** decision (negation must survive) | not migrating the auth service |
| Numbers that must stay exact | 340 ms, 200 ms, 15 percent, version 2.4 |

> When you run this sample through the app, use meeting context:
> `Kubernetes, PyTorch, ONNX, Kafka, Grafana, CI/CD, TensorRT, Priya, Arjun, Meera`

---

**Meera:** Okay, let's start. This is the weekly sync for the inference platform. Two things today: the
Kubernetes migration and the latency problem on the recommendation model.

**Arjun:** Sure. So the migration is mostly done. All services except auth are on the new Kubernetes
cluster. The CI/CD pipeline is deploying to it automatically since Monday.

**Meera:** Great. What about the auth service?

**Arjun:** Honestly, I don't think we should move auth this quarter. It's stable and the risk is high.

**Priya:** I agree. It's not worth it right now.

**Meera:** Okay, then it's decided — we are not migrating the auth service this quarter.

**Meera:** Next, latency. Priya, where are we?

**Priya:** The p95 latency for the recommendation model is at 340 milliseconds. Our target is 200. Most
of the time goes into the PyTorch model itself. I exported it to ONNX last week and in my tests that
alone cut latency by about 15 percent.

**Arjun:** Maybe we should also try TensorRT? That could be a lot faster.

**Meera:** Possibly, but let's not commit to that yet. Let's see the ONNX numbers in production first.

**Meera:** So, agreed: we roll out the ONNX version of the model with release 2.4.

**Priya:** Yes. Agreed.

**Meera:** Priya, can you put the latency numbers on a Grafana dashboard by Thursday?

**Priya:** Sure, I'll have it ready by Thursday.

**Meera:** Arjun, can you look into the Kafka consumer lag? It spiked again yesterday.

**Arjun:** Yeah, I'll take that.

**Meera:** We also need someone to write a load-test script before the 2.4 release.

**Arjun:** Hmm. Let's figure out who does that later.

**Priya:** One more thing — the GPU budget for next quarter. Do we ask for more?

**Meera:** Let's leave that for next week's meeting when finance joins. Okay, that's it. Thanks, everyone.

---

## What a correct output looks like

- **Decisions (2):** not migrating the auth service this quarter; rolling out the ONNX model with release 2.4.
- **Action items (3):**
  - Build a Grafana latency dashboard — **Priya**, **by Thursday**
  - Investigate Kafka consumer lag — **Arjun**, deadline *Unspecified*
  - Write a load-test script before the 2.4 release — owner *Unspecified*, deadline "before the 2.4 release"
- **Not decisions (open items):** trying TensorRT; GPU budget (deferred to next week).
