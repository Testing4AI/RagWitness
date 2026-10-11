"use strict";
const scenarios = window.RAGWITNESS_DEMO.scenarios;
const $ = id => document.getElementById(id);
let selected = 0, position = 0, timer = null;
const labels = ["Initial build", "Warm cache", "Update to v2", "Ask owner", "Rollback to v1", "Ask again"];
const explanations = [
  "The initial policy, index and fresh reference agree. The service fee is $10.",
  "The owner query fills the cache. The answer is Atlas; no knowledge has changed yet.",
  "The update has completed. The logical policy now says $20. Does every maintained representation agree?",
  "The owner is still Atlas. A matching answer does not establish that the new policy was installed correctly.",
  "Rollback restores v1. The current state can agree again—but an earlier completed update still failed.",
  "At the endpoint, the current state agrees. The history ledger retains any failure observed along the way."
];
function stop(){if(timer !== null) clearInterval(timer); timer=null; $("play").textContent="▶ Play history";}
function stateHTML(title,chunks){return `<strong>${title}</strong>`+chunks.map(c=>`<div class="chunk"><span>Policy · v${c.version}</span><b>Fee ${c.fee}</b></div>`).join("");}
function render(){
 const s=scenarios[selected], c=s.snapshots[position], hasFailure=c.ledger.length>0, bad=c.signatures.length>0;
 $("timeline").replaceChildren(...labels.map((label,i)=>{const b=document.createElement("button");b.innerHTML=`<small>${i===0?"START":"CHECKPOINT "+i}</small>${label}`;b.className=(i===position?"active ":"")+(i<=position&&s.snapshots[i].signatures.length?"seen":"");b.setAttribute("aria-current",i===position?"step":"false");b.onclick=()=>{stop();position=i;render();};return b;}));
 $("explanation").textContent=selected===3&&position>=4?"Clean execution: every completed operation remained consistent. Neither the current state nor the history ledger is flagged.":explanations[position];
 $("source").textContent=`Policy v${c.source_version} · fee ${c.expected[0].fee}`;
 $("actual").innerHTML=stateHTML("Maintained index",c.actual);$("actual").className="statebox"+(JSON.stringify(c.actual)!==JSON.stringify(c.expected)?" bad":"");
 $("expected").innerHTML=stateHTML("Fresh reference · same prescribed knowledge",c.expected);
 $("cache").textContent=c.cache_versions.length?"Cached evidence: "+c.cache_versions.map(v=>"v"+v).join(", "):"Empty / invalidated";
 $("cache").parentElement.className="cache"+(c.signatures.some(s=>s.layer==="cache")?" bad":"");
 $("answer").textContent=c.answer;
 $("verdict").textContent=bad?"Mismatch at this completed checkpoint":"Current state agrees with the reference";$("verdict").className="verdict"+(bad?" bad":"");
 $("answer-check").textContent="Same answer";$("state-check").textContent=bad?"MISMATCH":"AGREES";$("ledger-check").textContent=hasFailure?"FAILURE RETAINED":"NO FAILURE";
 if(hasFailure&&s.witness){const w=s.witness;$("witness").innerHTML=`<div class="count">${w.original_events} → ${w.minimized_events} <small>events</small></div><ol>${w.minimized_trace.map(e=>`<li>${e.action.replaceAll("_"," ")}${e.document?" · policy v"+e.document.version:""}</li>`).join("")}</ol>`;$("signature").innerHTML=`<span>PRESERVED FAILURE SIGNATURE</span>${w.signature.layer} / ${w.signature.document_id}<br><b>${w.signature.mismatch_kind}</b>`;}
 else{$("witness").innerHTML="<p>Waiting for a completed checkpoint with an observed failure.</p>";$("signature").textContent=selected===3?"Clean executions do not produce failure witnesses.":"The witness appears when a failure is observed.";}
 $("takeaway").textContent=selected===3?"A clean history stays clean. Failure witnesses are produced only when the declared consistency relation is violated.":position>=4?"A passing endpoint cannot erase a failed update. Keep the checkpoint evidence, then replay a small witness to investigate or regression-test a repair.":hasFailure?"The same answer can hide a different state. RAGWitness identifies the maintained representation that no longer matches the prescribed knowledge.":"Replay the update to see why testing the stateful software process complements testing its answers.";
 $("next").disabled=position===5;
}
$("scenario").onchange=e=>{stop();selected=Number(e.target.value);position=0;render();};
$("reset").onclick=()=>{stop();position=0;render();};
$("next").onclick=()=>{stop();position=Math.min(5,position+1);render();};
$("play").onclick=()=>{if(timer!==null){stop();return;}if(position===5)position=0;render();$("play").textContent="Ⅱ Pause";timer=setInterval(()=>{position++;render();if(position===5)stop();},2000);};
render();
