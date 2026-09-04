"use strict";

const camera = document.getElementById("camera");
const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");
const cameraStatus = document.getElementById("cameraStatus");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const recognitionStatus = document.getElementById("recognitionStatus");

// Only the three animals we currently support.
const animalAudio = {
  crab: document.getElementById("crabAudio"),
  snake: document.getElementById("snakeAudio"),
  dog: document.getElementById("dogAudio"),
};

let activeStream = null;
let imageModel = null;
let modelState = "loading";

let recognitionFrameId = null;
let lastPredictionTime = 0;
let recognitionSession = 0;

// Prediction smoothing
let smoothedScores = new Map();
let candidateLabel = null;
let candidateFrames = 0;

// Currently playing animal
let activeAnimal = null;
let audioSession = 0;

const MODEL_BASE_URL =
  "https://teachablemachine.withgoogle.com/models/2mx-T-4rY/";

// These MUST match the classes in your Teachable Machine model.
const SUPPORTED_CLASSES = new Set([
  "crab",
  "snake",
  "dog",
  "nothing",
]);

const MINIMUM_CONFIDENCE = 0.80;
const SMOOTHING_FACTOR = 0.35;
const REQUIRED_STABLE_PREDICTIONS = 4;
const PREDICTION_INTERVAL_MS = 160;
const ANIMAL_SOUND_THRESHOLD = 0.80;


// --------------------------------------------------
// CAMERA STATUS
// --------------------------------------------------

function setCameraStatus(message, state = "off") {
  cameraStatus.classList.remove("is-active", "is-error");

  if (state === "active") {
    cameraStatus.classList.add("is-active");
  }

  if (state === "error") {
    cameraStatus.classList.add("is-error");
  }

  cameraStatus.innerHTML =
    '<span class="camera-status__dot" aria-hidden="true"></span>' +
    message;
}


// --------------------------------------------------
// BUTTONS
// --------------------------------------------------

function updateControls(isRunning) {
  startButton.disabled = isRunning;
  stopButton.disabled = !isRunning;
}


// --------------------------------------------------
// RECOGNITION STATUS
// --------------------------------------------------

function setRecognitionStatus(message, state = "loading") {
  recognitionStatus.classList.remove(
    "is-ready",
    "is-detected",
    "is-error"
  );

  if (state === "ready") {
    recognitionStatus.classList.add("is-ready");
  }

  if (state === "detected") {
    recognitionStatus.classList.add("is-detected");
  }

  if (state === "error") {
    recognitionStatus.classList.add("is-error");
  }

  const dot = document.createElement("span");
  dot.className = "recognition-status__dot";
  dot.setAttribute("aria-hidden", "true");

  recognitionStatus.replaceChildren(
    dot,
    document.createTextNode(message)
  );
}


// --------------------------------------------------
// RESET PREDICTION SMOOTHING
// --------------------------------------------------

function resetRecognitionSmoothing() {
  smoothedScores = new Map();
  candidateLabel = null;
  candidateFrames = 0;
}


// --------------------------------------------------
// STOP ALL ANIMAL SOUNDS
// --------------------------------------------------

function stopAnimalSound() {
  audioSession += 1;
  activeAnimal = null;

  Object.values(animalAudio).forEach((audio) => {
    if (!audio) return;

    audio.pause();

    try {
      audio.currentTime = 0;
    } catch (error) {
      // Ignore audio reset errors.
    }
  });
}


// --------------------------------------------------
// START ANIMAL SOUND
// --------------------------------------------------

async function startAnimalSound(animal) {
  const audio = animalAudio[animal];

  if (!audio) {
    return;
  }

  // Already playing the correct sound.
  if (activeAnimal === animal && !audio.paused) {
    return;
  }

  // Stop whatever was playing first.
  stopAnimalSound();

  activeAnimal = animal;

  const session = ++audioSession;

  audio.loop = true;
  audio.currentTime = 0;

  try {
    await audio.play();

    // If the user changed animals while play() was starting,
    // immediately stop this old sound.
    if (
      activeAnimal !== animal ||
      session !== audioSession
    ) {
      audio.pause();

      try {
        audio.currentTime = 0;
      } catch (error) {
        // Ignore reset errors.
      }
    }
  } catch (error) {
    // Browser may block audio until the user interacts with the page.
    // Do not let this break recognition.
    if (
      activeAnimal === animal &&
      session === audioSession
    ) {
      activeAnimal = null;
    }

    console.warn(
      `Could not play ${animal} sound.`,
      error
    );
  }
}


// --------------------------------------------------
// UPDATE ANIMAL SOUND
// --------------------------------------------------

function updateAnimalSound(animal, probability) {
  const shouldPlay =
    animal &&
    animal !== "nothing" &&
    probability >= ANIMAL_SOUND_THRESHOLD;

  // No clear animal = stop sound.
  if (!shouldPlay) {
    if (activeAnimal !== null) {
      stopAnimalSound();
    }

    return;
  }

  // Start only if the detected animal changed.
  if (activeAnimal !== animal) {
    startAnimalSound(animal);
  }
}


// --------------------------------------------------
// LOAD TEACHABLE MACHINE MODEL
// --------------------------------------------------

async function loadAnimalModel() {
  try {
    if (!window.tmImage) {
      throw new Error(
        "Teachable Machine image library did not load"
      );
    }

    imageModel = await window.tmImage.load(
      `${MODEL_BASE_URL}model.json`,
      `${MODEL_BASE_URL}metadata.json`
    );

    modelState = "ready";

    if (activeStream) {
      startRecognition();
    } else {
      setRecognitionStatus(
        "Animal model ready — start the camera",
        "ready"
      );
    }
  } catch (error) {
    console.error("Model loading error:", error);

    imageModel = null;
    modelState = "error";

    setRecognitionStatus(
      "Animal model could not load",
      "error"
    );
  }
}


// --------------------------------------------------
// GET SMOOTHED TOP PREDICTION
// --------------------------------------------------

function getSmoothedTopPrediction(predictions) {
  const currentScores = new Map();

  // Only use the classes we actually want.
  predictions.forEach((prediction) => {
    const label = prediction.className
      .trim()
      .toLowerCase();

    if (!SUPPORTED_CLASSES.has(label)) {
      return;
    }

    currentScores.set(label, prediction.probability);
  });

  // Smooth EVERY supported class.
  // Missing classes are treated as zero.
  SUPPORTED_CLASSES.forEach((label) => {
    const currentScore = currentScores.get(label) || 0;
    const previousScore = smoothedScores.get(label) || 0;

    const nextScore =
      previousScore +
      SMOOTHING_FACTOR *
        (currentScore - previousScore);

    smoothedScores.set(label, nextScore);
  });

  const sorted = [...smoothedScores.entries()]
    .map(([label, probability]) => ({
      label,
      probability,
    }))
    .sort(
      (first, second) =>
        second.probability - first.probability
    );

  return sorted[0] || null;
}


// --------------------------------------------------
// STABLE PREDICTION
// --------------------------------------------------

function updateStablePrediction(prediction) {
  if (
    !prediction ||
    prediction.probability < MINIMUM_CONFIDENCE
  ) {
    candidateLabel = null;
    candidateFrames = 0;

    // Make absolutely sure no sound continues
    // when confidence falls below 80%.
    updateAnimalSound(null, 0);

    setRecognitionStatus(
      "Looking for a clear animal shadow…",
      "ready"
    );

    return;
  }

  // "Nothing" means no animal is currently visible.
  if (prediction.label === "nothing") {
    candidateLabel = null;
    candidateFrames = 0;

    updateAnimalSound(null, 0);

    setRecognitionStatus(
      "No animal detected",
      "ready"
    );

    return;
  }

  // Count consecutive frames with the same animal.
  if (prediction.label === candidateLabel) {
    candidateFrames += 1;
  } else {
    candidateLabel = prediction.label;
    candidateFrames = 1;
  }

  // Wait for a few consistent predictions
  // before showing "Detected".
  if (candidateFrames >= REQUIRED_STABLE_PREDICTIONS) {
    const confidence = Math.round(
      prediction.probability * 100
    );

    setRecognitionStatus(
      `Detected: ${prediction.label} (${confidence}%)`,
      "detected"
    );
  } else {
    setRecognitionStatus(
      "Checking the shadow…",
      "ready"
    );
  }
}


// --------------------------------------------------
// RECOGNITION LOOP
// --------------------------------------------------

async function runRecognition(timestamp, session) {
  if (
    !activeStream ||
    !imageModel ||
    session !== recognitionSession
  ) {
    return;
  }

  // Don't run the model too frequently.
  if (
    timestamp - lastPredictionTime <
    PREDICTION_INTERVAL_MS
  ) {
    recognitionFrameId = requestAnimationFrame(
      (nextTimestamp) =>
        runRecognition(nextTimestamp, session)
    );

    return;
  }

  // Wait until camera has usable video data.
  if (
    camera.readyState <
    HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    recognitionFrameId = requestAnimationFrame(
      (nextTimestamp) =>
        runRecognition(nextTimestamp, session)
    );

    return;
  }

  lastPredictionTime = timestamp;

  try {
    const predictions =
      await imageModel.predict(camera);

    // Camera/session may have changed while prediction
    // was running.
    if (
      !activeStream ||
      session !== recognitionSession
    ) {
      return;
    }

    const topPrediction =
      getSmoothedTopPrediction(predictions);

    updateStablePrediction(topPrediction);

    // Sound follows the actual confidence.
    if (
      topPrediction &&
      topPrediction.label !== "nothing" &&
      topPrediction.probability >=
        ANIMAL_SOUND_THRESHOLD
    ) {
      updateAnimalSound(
        topPrediction.label,
        topPrediction.probability
      );
    } else {
      updateAnimalSound(null, 0);
    }
  } catch (error) {
    console.error("Recognition error:", error);

    if (session === recognitionSession) {
      stopAnimalSound();

      setRecognitionStatus(
        "Recognition paused — please restart the camera",
        "error"
      );
    }

    return;
  }

  recognitionFrameId = requestAnimationFrame(
    (nextTimestamp) =>
      runRecognition(nextTimestamp, session)
  );
}


// --------------------------------------------------
// START RECOGNITION
// --------------------------------------------------

function startRecognition() {
  if (!activeStream) {
    return;
  }

  if (!imageModel) {
    if (modelState === "loading") {
      setRecognitionStatus(
        "Loading animal model…"
      );
    }

    return;
  }

  cancelAnimationFrame(recognitionFrameId);

  recognitionSession += 1;

  const session = recognitionSession;

  lastPredictionTime = 0;

  resetRecognitionSmoothing();

  // IMPORTANT:
  // Camera starting never starts an animal sound.
  stopAnimalSound();

  setRecognitionStatus(
    "Looking for a clear animal shadow…",
    "ready"
  );

  recognitionFrameId =
    requestAnimationFrame(
      (timestamp) =>
        runRecognition(timestamp, session)
    );
}


// --------------------------------------------------
// STOP RECOGNITION
// --------------------------------------------------

function stopRecognition() {
  recognitionSession += 1;

  cancelAnimationFrame(recognitionFrameId);

  recognitionFrameId = null;

  resetRecognitionSmoothing();

  if (modelState === "ready") {
    setRecognitionStatus(
      "Animal model ready — start the camera",
      "ready"
    );
  }
}


// --------------------------------------------------
// CAMERA ERROR MESSAGES
// --------------------------------------------------

function cameraErrorMessage(error) {
  switch (error?.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied";

    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera was found";

    case "NotReadableError":
    case "TrackStartError":
      return "Camera is busy in another app";

    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return "Requested camera is not available";

    default:
      return "Camera could not be started";
  }
}


// --------------------------------------------------
// REQUEST CAMERA
// --------------------------------------------------

async function requestCamera() {
  // First try the rear/environment camera.
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,

      video: {
        facingMode: {
          ideal: "environment",
        },
      },
    });
  } catch (error) {
    // If the requested camera is unavailable,
    // fall back to any available camera.
    if (
      error?.name === "OverconstrainedError" ||
      error?.name ===
        "ConstraintNotSatisfiedError" ||
      error?.name === "NotFoundError"
    ) {
      return navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }

    throw error;
  }
}


// --------------------------------------------------
// START CAMERA
// --------------------------------------------------

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus(
      "Camera is not supported",
      "error"
    );

    return;
  }

  if (!window.isSecureContext) {
    setCameraStatus(
      "Camera needs an HTTPS page",
      "error"
    );

    return;
  }

  if (activeStream) {
    stopCamera();
  }

  startButton.disabled = true;

  setCameraStatus(
    "Starting camera…"
  );

  try {
    activeStream = await requestCamera();

    camera.srcObject = activeStream;

    await camera.play();

    cameraPlaceholder.hidden = true;

    updateControls(true);

    const activeTrack =
      activeStream.getVideoTracks()[0];

    const facingMode =
      activeTrack?.getSettings?.().facingMode;

    setCameraStatus(
      facingMode === "environment"
        ? "Rear camera is on"
        : "Camera is on",
      "active"
    );

    startRecognition();
  } catch (error) {
    console.error("Camera error:", error);

    if (activeStream) {
      activeStream
        .getTracks()
        .forEach((track) => track.stop());
    }

    stopAnimalSound();

    activeStream = null;

    camera.srcObject = null;

    cameraPlaceholder.hidden = false;

    updateControls(false);

    setCameraStatus(
      cameraErrorMessage(error),
      "error"
    );
  }
}


// --------------------------------------------------
// STOP CAMERA
// --------------------------------------------------

function stopCamera() {
  stopRecognition();

  stopAnimalSound();

  if (activeStream) {
    activeStream
      .getTracks()
      .forEach((track) => track.stop());
  }

  activeStream = null;

  camera.srcObject = null;

  cameraPlaceholder.hidden = false;

  updateControls(false);

  setCameraStatus(
    "Camera stopped"
  );
}


// --------------------------------------------------
// BUTTON EVENTS
// --------------------------------------------------

startButton.addEventListener(
  "click",
  startCamera
);

stopButton.addEventListener(
  "click",
  stopCamera
);


// --------------------------------------------------
// CLEAN UP WHEN PAGE CLOSES
// --------------------------------------------------

window.addEventListener(
  "beforeunload",
  stopCamera
);


// --------------------------------------------------
// LOAD MODEL
// --------------------------------------------------

loadAnimalModel();