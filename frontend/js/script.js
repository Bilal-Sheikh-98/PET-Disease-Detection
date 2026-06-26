// ==============================
// API URL
// ==============================

const API_URL = "http://127.0.0.1:5000";

// ==============================
// IMAGE PREVIEW
// ==============================

function previewImage() {

    const file = document.getElementById("imageInput").files[0];

    if (!file) return;

    const preview = document.getElementById("preview");

    preview.src = URL.createObjectURL(file);

    preview.classList.remove("d-none");
}


// ==============================
// PREDICT
// ==============================

async function predict() {

    const file = document.getElementById("imageInput").files[0];

    if (!file) {

    Swal.fire({
        icon: "warning",
        title: "No Image Selected",
        text: "Please upload a Cat or Dog image before clicking Analyze Image.",
        confirmButtonColor: "#2563eb"
    });

    return;
}

    document.getElementById("loader").classList.remove("d-none");

    const fd = new FormData();

    fd.append("image", file);

    try {

        const response = await fetch(`${API_URL}/predict`, {

            method: "POST",

            body: fd

        });

        const data = await response.json();

        console.log("Prediction:", data);

        document.getElementById("animal").innerText = data.animal;

        document.getElementById("problem").innerText = data.problem;

        document.getElementById("advice").innerText = data.advice;

        document.getElementById("confidence").innerText =
            data.problem_confidence;

        document.getElementById("confidenceBar").style.width =
            data.problem_confidence + "%";

    }

    catch (err) {

        console.error(err);

        alert("Backend connection failed.");

    }

    finally {

        document.getElementById("loader").classList.add("d-none");

    }

}



// ==============================
// CURRENT LOCATION
// ==============================

function findClinics() {

    if (!navigator.geolocation) {

        alert("Geolocation not supported.");

        return;

    }

    navigator.geolocation.getCurrentPosition(

        successLocation,

        errorLocation

    );

}

function successLocation(position) {

    const lat = position.coords.latitude;

    const lon = position.coords.longitude;

    loadClinics(lat, lon);

}

function errorLocation() {

    alert("Please allow location permission.");

}



// ==============================
// LOAD CLINICS
// ==============================

async function loadClinics(lat, lon) {

    try {

        const response = await fetch(

            `${API_URL}/nearby-clinics?lat=${lat}&lon=${lon}`

        );

        const data = await response.json();

        console.log(data);

        const clinicDiv = document.getElementById("clinics");

        clinicDiv.innerHTML = "";

        data.clinics.forEach((clinic, index) => {

            clinicDiv.innerHTML += `

<div class="clinic-card">

<h5>

<i class="fa-solid fa-hospital text-success"></i>

${clinic.name}

</h5>

<p>

📍 Distance:

<b>${clinic.distance_km} km</b>

</p>
<p>
    <i class="fa-solid fa-location-dot text-danger"></i>
    ${clinic.address}
</p>
<p>
    📞 <b>${clinic.phone}</b>
</p>
<div class="d-flex gap-2 mt-3">

<button

class="btn btn-primary btn-sm"
onclick="showMap(${clinic.latitude}, ${clinic.longitude})">

View on Map

</button>

<a

class="btn btn-success btn-sm"

target="_blank"

href="${clinic.map_link}">

Open Google Maps

</a>

</div>

</div>

`;

            if (index === 0) {

                // showMap(clinic.map_link);
                showMap(clinic.latitude, clinic.longitude);

            }

        });

    }

    catch (err) {

        console.error(err);

    }

}



// ==============================
// SHOW MAP
// ==============================
function showMap(lat, lon) {

    document.getElementById("mapFrame").src =
        `https://maps.google.com/maps?q=${lat},${lon}&z=16&output=embed`;

}
// function showMap(link) {

//     const coordinates = link.split("=")[1];

//     document.getElementById("mapFrame").src =

//         `https://maps.google.com/maps?q=${coordinates}&z=15&output=embed`;

// }



// ==============================
// AUTO FIND LOCATION
// ==============================

// window.onload = function () {

//     findClinics();

// };