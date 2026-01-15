function autofill() {
  document.getElementById("GrLivArea").value = 1800;
  document.getElementById("OverallQual").value = 7;
  document.getElementById("GarageCars").value = 2;
}

async function predict() {
  const data = {
    GrLivArea: Number(GrLivArea.value),
    OverallQual: Number(OverallQual.value),
    GarageCars: Number(GarageCars.value)
  };

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });

  const result = await res.json();

  document.getElementById("result").innerHTML =
    `Estimated Price: <b>$${result.prediction}</b>
     <br>CI: $${result.lower} – $${result.upper}`;

  document.querySelector(".bar").style.width = "100%";
}
