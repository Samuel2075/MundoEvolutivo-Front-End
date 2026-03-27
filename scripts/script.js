const canvas = document.getElementById('world');
const ctx = canvas.getContext('2d', { alpha: false });
const TILE_SIZE = 28;
const MAX_DPR = 1.5;
const TERRAIN_CACHE_LIMIT = 140;
const HUD_UPDATE_INTERVAL_MS = 120;
const OVERVIEW_SCALE = 0.125;
const OVERVIEW_ZOOM_THRESHOLD = 0.18;
const RESOURCE_DRAW_ZOOM_THRESHOLD = 0.14;
const ANIMAL_DRAW_ZOOM_THRESHOLD = 0.16;
const CARCASS_DRAW_ZOOM_THRESHOLD = 0.14;
const WORLD_COLS = 500;
const WORLD_ROWS = 500;
const WORLD_WIDTH = WORLD_COLS * TILE_SIZE;
const WORLD_HEIGHT = WORLD_ROWS * TILE_SIZE;
const SIM_TICK_MS = 140;
const ENTITY_LABEL_ZOOM_THRESHOLD = 0.2;
const HUMAN_DRAW_ZOOM_THRESHOLD = 0.14;
const HUMAN_LEGEND_UPDATE_INTERVAL_MS = 700;
const HUMAN_ACTION_LIBRARY = [
    'beber',
    'comer',
    'explorar',
    'caçar',
    'pescar',
    'coletar-recurso',
    'construir',
    'armazenar',
    'descansar',
    'acasalar',
    'ajudar',
    'socializar',
    'fugir',
    'reparar-base',
    'reparar-arma',
    'trocar',
    'atacar',
    'craftar'
];
const ACTIONS = {
    beber: drinkWater,
    comer: eatFood,
    explorar: explore,
    caçar: hunt,
    pescar: fish,
    'coletar-recurso': gatherResource,
    construir: buildStructure,
    armazenar: storeItem,
    descansar: rest,
    acasalar: reproduce,
    ajudar: helpOther,
    socializar: socialize,
    fugir: flee,
    'reparar-base': repairBase,
    trocar: trade,
    atacar: attack,
    'craftar': craftItem
};
const CRAFT_RECIPES = {
    lança: { wood: 2, stone: 1, metal: 0, reed: 0 },
    espada: { wood: 1, stone: 0, metal: 3, reed: 0 },
    escudo: { wood: 3, stone: 2, metal: 0, reed: 0 },
    armadura: { wood: 0, stone: 0, metal: 4, reed: 2 },
    tocha: { wood: 2, stone: 0, metal: 0, reed: 1 },
};

// bônus que cada item concede
const CRAFT_EFFECTS = {
    lança: { attackBonus: 15, defenseBonus: 0, visionBonus: 0, lightBonus: 0 },
    espada: { attackBonus: 30, defenseBonus: 0, visionBonus: 0, lightBonus: 0 },
    escudo: { attackBonus: 0, defenseBonus: 25, visionBonus: 0, lightBonus: 0 },
    armadura: { attackBonus: 0, defenseBonus: 40, visionBonus: 0, lightBonus: 0 },
    tocha: { attackBonus: 0, defenseBonus: 0, visionBonus: 80, lightBonus: 1 },
};
const HUMAN_ALLOWED_BIOMES = ['plains', 'forest', 'swamp', 'mountain'];
const worldSizeLabel = document.getElementById('worldSize');
const resourceCountLabel = document.getElementById('resourceCount');
const focusBiomeLabel = document.getElementById('focusBiome');
const cameraInfoLabel = document.getElementById('cameraInfo');
const animalCountLabel = document.getElementById('animalCountLabel');
const carcassCountLabel = document.getElementById('carcassCountLabel');
const birthBudgetLabel = document.getElementById('birthBudgetLabel');
const extinctSpeciesLabel = document.getElementById('extinctSpeciesLabel');
const faunaLegend = document.getElementById('faunaLegend');
const humanCountLabel = document.getElementById('humanCountLabel');
const humanGenerationLabel = document.getElementById('humanGenerationLabel');
const humanWisdomLabel = document.getElementById('humanWisdomLabel');
const humanEfficiencyLabel = document.getElementById('humanEfficiencyLabel');
const humanBaseLabel = document.getElementById('humanBaseLabel');
const humanActionLabel = document.getElementById('humanActionLabel');
const humanTopScoreLabel = document.getElementById('humanTopScoreLabel');
const humanLegend = document.getElementById('humanLegend');
const resourceGridRecursos = document.getElementById('resource-grid-recursos');
const zoomInButton = document.getElementById('zoomInButton');
const zoomOutButton = document.getElementById('zoomOutButton');
const pauseButton = document.getElementById('pauseButton');
const footerTip = document.getElementById('footerTip');

let savedGenerationWeights = null; // pesos herdados da geração anterior
let cycleCount = 0;                // contador de ciclos

function createHumanBrain() {

    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [10],
        units: 32,
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: 32,
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: HUMAN_ACTION_LIBRARY.length,
        activation: "softmax"
    }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy"
    });

    return model;
}

function countInInventory(human, type) {
    return human.inventory.filter(i => i.type === type).length;
}

function removeFromInventory(human, type, amount) {
    let removed = 0;
    human.inventory = human.inventory.filter(i => {
        if (i.type === type && removed < amount) {
            removed++;
            return false;
        }
        return true;
    });
}

function humanStateVector(human) {

    const hunger = human.needs?.hunger || 0;
    const thirst = human.needs?.thirst || 0;
    const health = human.genes?.health || 0;

    const wood = human.inventory.filter(i => i.type === "wood").length;
    const stone = human.inventory.filter(i => i.type === "stone").length;
    const food = human.inventory.filter(i => i.type === "food").length;

    const hasBase = human.baseId ? 1 : 0;

    const nearHuman = findNearestHuman(human, 120) ? 1 : 0;
    const nearAnimal = findNearestAnimal(human, 120) ? 1 : 0;

    const energy = human.energy || 0;

    return [
        hunger / 100,
        thirst / 100,
        health / 100,
        wood / 10,
        stone / 10,
        food / 10,
        hasBase,
        nearHuman,
        nearAnimal,
        energy / 100
    ];
}

function chooseAllHumanActions() {
    const alive = humans.filter(h => h.alive);
    if (!alive.length) return;

    for (const human of alive) {
        const inputTensor = tf.tensor2d([humanStateVector(human)]);
        const prediction = human.brain.predict(inputTensor);
        const probs = prediction.dataSync();

        inputTensor.dispose();
        prediction.dispose();

        let bestIndex = 0;
        for (let j = 1; j < probs.length; j++) {
            if (probs[j] > probs[bestIndex]) bestIndex = j;
        }

        human.pendingAction = HUMAN_ACTION_LIBRARY[bestIndex];
    }
}

function rewardHuman(human, value) {

    if (!human.memory || !human.memory.length) return;
    const last = human.memory[human.memory.length - 1];
    last.reward = (last.reward || 0) + value;
}

let isTraining = false;

setInterval(async () => {
    if (isTraining || !simRunning) return;
    isTraining = true;
    try {
        for (const human of humans.filter(h => h.alive)) {
            if (!human.memory || human.memory.length < 10) continue;

            const batch = [];
            for (let i = 0; i < 32; i++) {
                batch.push(human.memory[Math.floor(Math.random() * human.memory.length)]);
            }

            const xs = tf.tensor2d(batch.map(e => e.state));
            const ys = tf.tensor2d(batch.map(e => {
                const label = new Array(HUMAN_ACTION_LIBRARY.length).fill(0.05);
                if (e.action >= 0) {
                    // normaliza entre 0 e 1 — recompensas negativas ficam abaixo de 0.5
                    label[e.action] = Math.max(0, Math.min(1, (e.reward + 1) / 2));
                }
                return label;
            }));

            await human.brain.fit(xs, ys, { epochs: 3, verbose: 0 });
            xs.dispose();
            ys.dispose();

            human.mlUpdateCounter = (human.mlUpdateCounter || 0) + 1;
            if (human.memory.length > 200) human.memory = human.memory.slice(-200);
        }
    } finally {
        isTraining = false;
    }
}, 1000);

function updateHumans(dtMs, now) {

    if (!humans.length) return;

    const dtSeconds = dtMs / 1000;

    for (const human of humans) {

        if (!human.alive) continue;

        // envelhecimento
        human.ageMs += dtMs;

        // cooldown decisão
        human.decisionCooldownMs -= dtMs;

        // morte
        if (human.genes.health <= 0) {
            human.alive = false;
            rewardHuman(human, -10);
            continue;
        }

        // estado fisiológico simples
        human.needs.hunger += dtSeconds * 0.3;
        human.needs.thirst += dtSeconds * 0.4;

        if (human.needs.hunger > 100 || human.needs.thirst > 100) {
            human.genes.health -= dtSeconds * 5;
        }

        // DECISÃO DA IA
        if (human.decisionCooldownMs <= 0) {

            const stateBefore = humanStateVector(human);
            const action = human.pendingAction || 'explorar';
            const actionFn = ACTIONS[action];

            if (actionFn) {
                actionFn(human);
            }

            human.lastAction = action;
            if (!human.memory) human.memory = [];

            human.memory.push({
                state: stateBefore,
                action: HUMAN_ACTION_LIBRARY.indexOf(action),
                reward: 0
            });

            human.decisionCooldownMs = 300 + Math.random() * 200;
        }

        if (isNearWorldEdge(human)) {
            rewardHuman(human, -10);

            // human.targetX = WORLD_WIDTH / 2 + (Math.random() - 0.5) * TILE_SIZE * 10;
            // human.targetY = WORLD_HEIGHT / 2 + (Math.random() - 0.5) * TILE_SIZE * 10;
        }

        if (human.ageMs >= HUMAN_LIFESPAN_MS) {
            human.alive = false;
            rewardHuman(human, 5); // recompensa por ter vivido até o fim
            spawnDeathToast(human, 'Morreu de velhice');
            continue;
        }

        // MOVIMENTO
        if (human.vx !== undefined) {

            const nextX = clamp(human.x + human.vx * dtSeconds, 0, WORLD_WIDTH);
            const nextY = clamp(human.y + human.vy * dtSeconds, 0, WORLD_HEIGHT);

            const tile = worldToTile(nextX, nextY);

            if (!isWaterTile(tile)) {
                human.x = nextX;
                human.y = nextY;
            }
        }

        if (!human.vx && !human.vy) {

            const dx = human.targetX - human.x;
            const dy = human.targetY - human.y;

            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < 5) {

                const angle = Math.random() * Math.PI * 2;
                const distMove = 120 + Math.random() * 200;

                human.targetX = clamp(human.x + Math.cos(angle) * distMove, 0, WORLD_WIDTH);
                human.targetY = clamp(human.y + Math.sin(angle) * distMove, 0, WORLD_HEIGHT);

            }

            const dx2 = human.targetX - human.x;
            const dy2 = human.targetY - human.y;

            const len = Math.sqrt(dx2 * dx2 + dy2 * dy2);

            if (len > 0) {
                const speed = 35 + human.genes.speed * 40;

                human.vx = (dx2 / len) * speed;
                human.vy = (dy2 / len) * speed;
            }

        }

        // recompensas naturais

        const hungerReward = (100 - human.needs.hunger) / 100;   // 0 a 1
        const thirstReward = (100 - human.needs.thirst) / 100;   // 0 a 1
        const healthReward = human.genes.health / 100;            // 0 a 1
        const energyReward = human.needs.energy / 100;            // 0 a 1
        const survivalScore = (hungerReward + thirstReward + healthReward + energyReward) / 4;
        rewardHuman(human, survivalScore * 0.5);

    }

    humans = humans.filter(h => h.alive);
}

function isWaterTile(tile) {
    return tile && tile.biome === "water";
}

function isMobileViewport() {
    return window.matchMedia('(max-width: 768px)').matches;
}

function updateMobileUIState() {
    if (isMobileViewport()) {
        document.querySelectorAll('.hud .collapse-panel, .legend .collapse-panel').forEach((panel, index) => {
            if (index > 0) panel.removeAttribute('open');
        });
    }
}

function zoomCameraAt(screenX, screenY, zoomFactor) {
    const pointBefore = screenToWorld(screenX, screenY);
    camera.zoom = clamp(camera.zoom * zoomFactor, 0.08, 2.4);
    const pointAfter = screenToWorld(screenX, screenY);
    camera.x = clamp(camera.x + (pointBefore.x - pointAfter.x), 0, WORLD_WIDTH);
    camera.y = clamp(camera.y + (pointBefore.y - pointAfter.y), 0, WORLD_HEIGHT);
    requestRender();
}

function focusCameraOnHuman(humanId) {
    const human = humans.find((item) => item.id === humanId && item.alive);
    if (!human) return;
    camera.x = clamp(human.x, 0, WORLD_WIDTH);
    camera.y = clamp(human.y, 0, WORLD_HEIGHT);
    if (camera.zoom < 0.6) camera.zoom = 0.6;
    requestRender();
}

humanLegend?.addEventListener('click', (event) => {
    const entry = event.target.closest('.human-focus-target');
    if (!entry) return;
    event.preventDefault();
    const humanId = Number(entry.dataset.humanId);
    if (!Number.isFinite(humanId)) return;
    focusCameraOnHuman(humanId);
});

const BIOMES = {
    plains: { name: 'Planície', color: '#67c16f', resourceDensity: 0.33, resources: ['berry', 'herb', 'water', 'berry'] },
    forest: { name: 'Floresta', color: '#2f7d32', resourceDensity: 0.42, resources: ['stone', 'wood', 'berry', 'mushroom', 'herb'] },
    desert: { name: 'Deserto', color: '#c2a36a', resourceDensity: 0.2, resources: ['cactus', 'crystal', 'water'] },
    mountain: { name: 'Montanha', color: '#8e6b3f', resourceDensity: 0.27, resources: ['stone', 'metal', 'stone', 'metal'] },
    swamp: { name: 'Pântano', color: '#5bc0de', resourceDensity: 0.36, resources: ['reed', 'medicinal', 'water', 'mushroom'] },
    lake: { name: 'Água', color: '#4f86f7', resourceDensity: 0.18, resources: ['fish', 'water'] },
    snow: { name: 'Neve', color: '#9bd4ff', resourceDensity: 0.22, resources: ['ice', 'moss', 'water', 'ice'] }
};

const RESOURCE_STYLE = {
    berry: { color: '#e63946', radius: 3, label: 'Fruta' },
    herb: { color: '#a7f3d0', radius: 2, label: 'Erva' },
    water: { color: '#60a5fa', radius: 4, label: 'Água' },
    wood: { color: '#6b4423', radius: 4, label: 'Madeira' },
    mushroom: { color: '#fca5a5', radius: 3, label: 'Cogumelo' },
    cactus: { color: '#2e8b57', radius: 4, label: 'Cacto' },
    crystal: { color: '#c084fc', radius: 3, label: 'Cristal' },
    stone: { color: '#9ca3af', radius: 4, label: 'Pedra' },
    metal: { color: '#cbd5e1', radius: 3, label: 'Metal' },
    reed: { color: '#84cc16', radius: 3, label: 'Junco' },
    medicinal: { color: '#22c55e', radius: 3, label: 'Planta medicinal' },
    fish: { color: '#93c5fd', radius: 3, label: 'Peixe' },
    ice: { color: '#dbeafe', radius: 3, label: 'Gelo' },
    moss: { color: '#86efac', radius: 2, label: 'Musgo' },
    meat: { color: '#b91c1c', radius: 5, label: 'Carne' }
};
const HUMAN_LIFESPAN_MS = 180000;
const htmlResourceRecurso = "";

resourceGridRecursos.append

const ANIMAL_SPECIES = [
    { id: 'rabbit', name: 'Coelho', color: '#f5d0a9', biomes: ['plains', 'forest', 'snow'], diet: 'herbivore', prey: [], speed: 72, size: 5, vision: 180, fleeRadius: 165, attackRange: 0, threat: 6, meatRange: [14, 22], initialPopulation: 30, populationCap: 100, respawnBudget: 285, lifespanMs: 170000 },
    { id: 'hare', name: 'Lebre-da-Neve', color: '#f8fafc', biomes: ['snow', 'plains', 'mountain'], diet: 'herbivore', prey: [], speed: 74, size: 5, vision: 185, fleeRadius: 170, attackRange: 0, threat: 7, meatRange: [13, 21], initialPopulation: 30, populationCap: 100, respawnBudget: 273, lifespanMs: 165000 },
    { id: 'deer', name: 'Cervo', color: '#c68642', biomes: ['plains', 'forest', 'swamp'], diet: 'herbivore', prey: [], speed: 64, size: 7, vision: 210, fleeRadius: 190, attackRange: 0, threat: 10, meatRange: [34, 56], initialPopulation: 30, populationCap: 100, respawnBudget: 225, lifespanMs: 240000 },
    { id: 'antelope', name: 'Antílope', color: '#d6b07d', biomes: ['plains', 'desert'], diet: 'herbivore', prey: [], speed: 70, size: 6, vision: 215, fleeRadius: 195, attackRange: 0, threat: 11, meatRange: [28, 44], initialPopulation: 30, populationCap: 100, respawnBudget: 237, lifespanMs: 220000 },
    { id: 'boar', name: 'Javali', color: '#7a4e2d', biomes: ['forest', 'swamp', 'plains'], diet: 'omnivore', prey: ['rabbit', 'hare', 'lizard'], speed: 56, size: 8, vision: 170, fleeRadius: 135, attackRange: 16, threat: 26, meatRange: [42, 64], initialPopulation: 30, populationCap: 100, respawnBudget: 165, lifespanMs: 230000 },
    { id: 'beaver', name: 'Castor', color: '#8b5e3c', biomes: ['forest', 'lake', 'swamp'], diet: 'herbivore', prey: [], speed: 38, size: 5, vision: 120, fleeRadius: 120, attackRange: 0, threat: 8, meatRange: [16, 26], initialPopulation: 30, populationCap: 100, respawnBudget: 192, lifespanMs: 210000 },
    { id: 'otter', name: 'Lontra', color: '#7c5c42', biomes: ['lake', 'swamp', 'forest'], diet: 'omnivore', prey: [], speed: 52, size: 5, vision: 145, fleeRadius: 130, attackRange: 0, threat: 12, meatRange: [15, 24], initialPopulation: 30, populationCap: 100, respawnBudget: 156, lifespanMs: 205000 },
    { id: 'fox', name: 'Raposa', color: '#ff8c42', biomes: ['forest', 'plains', 'desert'], diet: 'carnivore', prey: ['rabbit', 'hare', 'lizard'], speed: 68, size: 6, vision: 205, fleeRadius: 150, attackRange: 16, threat: 34, meatRange: [18, 30], initialPopulation: 30, populationCap: 100, respawnBudget: 126, lifespanMs: 210000 },
    { id: 'fennec', name: 'Raposa-do-Deserto', color: '#f0c27a', biomes: ['desert'], diet: 'carnivore', prey: ['lizard', 'rabbit'], speed: 66, size: 5, vision: 198, fleeRadius: 150, attackRange: 14, threat: 29, meatRange: [14, 24], initialPopulation: 30, populationCap: 100, respawnBudget: 120, lifespanMs: 195000 },
    { id: 'wolf', name: 'Lobo', color: '#bcc7d1', biomes: ['forest', 'snow', 'plains'], diet: 'carnivore', prey: ['rabbit', 'hare', 'deer', 'fox', 'goat', 'antelope'], speed: 76, size: 7, vision: 250, fleeRadius: 120, attackRange: 18, threat: 55, meatRange: [24, 38], initialPopulation: 30, populationCap: 100, respawnBudget: 108, lifespanMs: 230000 },
    { id: 'lynx', name: 'Lince', color: '#b7794b', biomes: ['forest', 'snow', 'mountain'], diet: 'carnivore', prey: ['rabbit', 'hare', 'fox', 'goat'], speed: 71, size: 6, vision: 225, fleeRadius: 115, attackRange: 17, threat: 48, meatRange: [20, 30], initialPopulation: 30, populationCap: 100, respawnBudget: 99, lifespanMs: 215000 },
    { id: 'bear', name: 'Urso', color: '#5b3723', biomes: ['forest', 'mountain', 'snow'], diet: 'omnivore', prey: ['rabbit', 'hare', 'deer', 'fox', 'boar', 'goat', 'beaver'], speed: 50, size: 10, vision: 215, fleeRadius: 100, attackRange: 20, threat: 74, meatRange: [70, 110], initialPopulation: 30, populationCap: 100, respawnBudget: 72, lifespanMs: 300000 },
    { id: 'moose', name: 'Alce', color: '#8d5b34', biomes: ['forest', 'swamp', 'snow'], diet: 'herbivore', prey: [], speed: 53, size: 9, vision: 185, fleeRadius: 165, attackRange: 0, threat: 18, meatRange: [60, 88], initialPopulation: 30, populationCap: 100, respawnBudget: 117, lifespanMs: 255000 },
    { id: 'buffalo', name: 'Búfalo', color: '#66503d', biomes: ['plains', 'swamp'], diet: 'herbivore', prey: [], speed: 49, size: 9, vision: 175, fleeRadius: 160, attackRange: 0, threat: 17, meatRange: [62, 92], initialPopulation: 30, populationCap: 100, respawnBudget: 123, lifespanMs: 250000 },
    { id: 'camel', name: 'Camelo', color: '#cfb07a', biomes: ['desert'], diet: 'herbivore', prey: [], speed: 58, size: 8, vision: 210, fleeRadius: 170, attackRange: 0, threat: 12, meatRange: [44, 62], initialPopulation: 30, populationCap: 100, respawnBudget: 150, lifespanMs: 250000 },
    { id: 'lizard', name: 'Lagarto', color: '#6daa4a', biomes: ['desert', 'mountain'], diet: 'omnivore', prey: [], speed: 48, size: 5, vision: 130, fleeRadius: 130, attackRange: 0, threat: 8, meatRange: [10, 18], initialPopulation: 30, populationCap: 100, respawnBudget: 186, lifespanMs: 180000 },
    { id: 'vulture', name: 'Abutre', color: '#7f8c8d', biomes: ['desert', 'mountain', 'swamp'], diet: 'carnivore', prey: [], speed: 62, size: 6, vision: 240, fleeRadius: 105, attackRange: 0, threat: 18, meatRange: [12, 20], initialPopulation: 30, populationCap: 100, respawnBudget: 81, lifespanMs: 220000 },
    { id: 'hyena', name: 'Hiena', color: '#c19a6b', biomes: ['plains', 'desert'], diet: 'carnivore', prey: ['rabbit', 'hare', 'antelope', 'fox'], speed: 63, size: 7, vision: 205, fleeRadius: 118, attackRange: 17, threat: 44, meatRange: [22, 34], initialPopulation: 30, populationCap: 100, respawnBudget: 93, lifespanMs: 225000 },
    { id: 'goat', name: 'Cabra-Montês', color: '#d8d1c3', biomes: ['mountain', 'plains', 'snow'], diet: 'herbivore', prey: [], speed: 61, size: 6, vision: 180, fleeRadius: 168, attackRange: 0, threat: 9, meatRange: [26, 38], initialPopulation: 30, populationCap: 100, respawnBudget: 174, lifespanMs: 215000 },
    { id: 'eagle', name: 'Águia', color: '#d4a373', biomes: ['mountain', 'forest', 'plains'], diet: 'carnivore', prey: ['rabbit', 'hare', 'lizard', 'fennec'], speed: 88, size: 5, vision: 285, fleeRadius: 108, attackRange: 14, threat: 41, meatRange: [12, 20], initialPopulation: 30, populationCap: 100, respawnBudget: 87, lifespanMs: 210000 },
    { id: 'yak', name: 'Iaque', color: '#3e2f25', biomes: ['mountain', 'snow'], diet: 'herbivore', prey: [], speed: 54, size: 9, vision: 175, fleeRadius: 165, attackRange: 0, threat: 16, meatRange: [58, 86], initialPopulation: 30, populationCap: 100, respawnBudget: 138, lifespanMs: 260000 },
    { id: 'crocodile', name: 'Crocodilo', color: '#4e7d46', biomes: ['swamp', 'lake'], diet: 'carnivore', prey: ['rabbit', 'deer', 'boar', 'otter', 'antelope'], speed: 42, size: 9, vision: 190, fleeRadius: 100, attackRange: 22, threat: 68, meatRange: [54, 84], initialPopulation: 30, populationCap: 100, respawnBudget: 84, lifespanMs: 270000 },
    { id: 'arcticFox', name: 'Raposa-Ártica', color: '#dfe7ef', biomes: ['snow', 'mountain'], diet: 'carnivore', prey: ['hare', 'rabbit', 'lizard'], speed: 69, size: 5, vision: 210, fleeRadius: 145, attackRange: 15, threat: 31, meatRange: [14, 22], initialPopulation: 30, populationCap: 100, respawnBudget: 105, lifespanMs: 205000 },
    { id: 'muskOx', name: 'Boi-Almiscarado', color: '#5c4635', biomes: ['snow', 'plains'], diet: 'herbivore', prey: [], speed: 46, size: 9, vision: 168, fleeRadius: 158, attackRange: 0, threat: 15, meatRange: [64, 94], initialPopulation: 30, populationCap: 100, respawnBudget: 111, lifespanMs: 265000 }
];
const SPECIES_MAP = Object.fromEntries(ANIMAL_SPECIES.map((species) => [species.id, species]));

let world = null;
let overviewTerrainCanvas = null;
let overviewResourcesCanvas = null;
const CHUNK_TILES = 24;
const terrainChunks = new Map();
const resourceChunks = new Map();
const animalChunks = new Map();
const humanChunks = new Map();
let needsRender = true;
let renderQueued = false;
let lastHudUpdate = 0;
let lastHumanLegendUpdate = 0;
let animals = [];
let carcasses = [];
let humans = [];
let bases = [];
let buildSites = [];
let initialHumanFocus = null;
let respawnQueue = [];
let speciesState = new Map();
let simulationRng = Math.random;
let animalIdCounter = 0;
let carcassIdCounter = 0;
let humanIdCounter = 0;
let baseIdCounter = 0;
let buildSiteIdCounter = 0;
let simRunning = true;
let simulationLoopId = null;
let simulationTimeMs = 0;
let totalBasesBuiltThisRun = 0;
let currentRunActionTotals = {};
let currentRunDeathRecords = [];
let currentRunChildrenBorn = 0;
let currentRunHumansBorn = 0;

let extinctionModalActive = false;
let currentCycleFinalized = false;
let currentCycleSummary = null;
let pendingCycleSummary = null;

let camera = {
    x: WORLD_WIDTH * 0.5,
    y: WORLD_HEIGHT * 0.5,
    zoom: 0.5
};

let drag = {
    active: false,
    startX: 0,
    startY: 0,
    cameraStartX: 0,
    cameraStartY: 0
};

let touchState = {
    active: false,
    mode: null,
    startDistance: 0,
    startZoom: 1,
    startCenterX: 0,
    startCenterY: 0,
    cameraStartX: 0,
    cameraStartY: 0,
    lastX: 0,
    lastY: 0
};

function renderResourceGrid() {
    const resourceGridRecursos = document.getElementById('resource-grid-recursos');

    resourceGridRecursos.innerHTML = '';

    Object.values(RESOURCE_STYLE).forEach(resource => {
        const div = document.createElement('div');
        div.className = 'resource-entry';

        div.innerHTML = `
      <span class="swatch" style="background:${resource.color}"></span>
      <span>${resource.label}</span>
    `;

        resourceGridRecursos.appendChild(div);
    });
}

function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    requestRender();
}

// INICIO - Funções que serão executadas a partir de uma ação
function buildActions(action, human) {
    const fn = ACTIONS[action];
    if (fn) fn(human);
}

function drinkWater(human) {
    const water = findNearestResource(human, TILE_SIZE * 2, new Set(['water']));

    if (!water) rewardHuman(human, -10);

    human.needs.thirst = Math.max(0, human.needs.thirst - 40);
    rewardHuman(human, human.needs.thirst > 60 ? 2.0 : 0.5);
}

function eatFood(human) {

    const foodIndex = human.inventory.findIndex(item =>
        ['berry', 'fish', 'meat', 'mushroom', 'cactus'].includes(item.type)
    );

    if (foodIndex === -1) rewardHuman(human, -10);

    human.inventory.splice(foodIndex, 1);

    human.hunger = Math.max(0, human.needs.hunger - 40);
    rewardHuman(human, human.needs.hunger > 60 ? 2.0 : 0.5);
}

function explore(human) {

    const angle = Math.random() * Math.PI * 2;
    const distance = 120;

    human.targetX = human.x + Math.cos(angle) * distance;
    human.targetY = human.y + Math.sin(angle) * distance;

}

function gatherResource(human) {

    const resource = findNearestResource(human, TILE_SIZE * 1.5);

    if (!resource) { rewardHuman(human, -10); return; }
    human.inventory.push({ type: resource.type, amount: 1 });

    removeResource(resource);
    rewardHuman(human, 0.8);
}

function fish(human) {

    const water = findNearestResource(human, TILE_SIZE * 1.5, new Set(['water']));

    if (!water) { rewardHuman(human, -10); return; }
    if (Math.random() < 0.4) {

        human.inventory.push({
            type: 'fish',
            amount: 1
        });

    }

}

function hunt(human) {

    const animal = findNearestAnimal(human, TILE_SIZE * 4);

    if (!animal) { rewardHuman(human, -10); return; }
    if (Math.random() < 0.5) {
        killAnimal(animal);

        human.inventory.push({
            type: 'meat',
            amount: 2
        });

    }
    rewardHuman(human, 1.5);
}

function rest(human) {
    human.vx = 0;
    human.vy = 0;
    human.targetX = human.x;
    human.targetY = human.y;

    human.needs.energy = Math.min(100, human.needs.energy + 25);
    rewardHuman(human, human.needs.energy < 30 ? 1.5 : 0.2);
}

function buildStructure(human) {

    const woodNeeded = 15;
    const stoneNeeded = 10;

    const wood = human.inventory.filter(i => i.type === "wood");
    const stone = human.inventory.filter(i => i.type === "stone");

    if (wood.length < woodNeeded || stone.length < stoneNeeded) { rewardHuman(human, -10); return; }
    createBase(human.x, human.y);

}

function storeItem(human) {

    const base = findNearestBase(human, TILE_SIZE * 2);

    if (!base) { rewardHuman(human, -10); return; }

    if (human.inventory.length === 0) rewardHuman(human, -10);

    human.vx = 0;
    human.vy = 0;
    human.targetX = human.x;
    human.targetY = human.y;

    const item = human.inventory.pop();

    base.storage.push(item);

}

function findNearestBase(human, radius, filterFn = null) {

    let nearest = null;
    let bestDistSq = radius * radius;

    for (const base of bases) {

        const distSq = distanceSq(human.x, human.y, base.x, base.y);

        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            nearest = base;
        }
    }

    return nearest;
}

function findNearestHuman(human, radius, filterFn = null) {

    let nearest = null;
    let bestDistSq = radius * radius;

    for (const other of humans) {

        if (!other.alive) continue;
        if (other.id === human.id) continue;

        if (filterFn && !filterFn(other)) continue;

        const distSq = distanceSq(human.x, human.y, other.x, other.y);

        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            nearest = other;
        }
    }

    return nearest;
}

function reproduce(human) {
    const partner = findNearestHuman(human, TILE_SIZE * 2,
        h => h.sex !== human.sex
    );

    if (!partner) return;

    const distance = randomBetween(simulationRng, TILE_SIZE * 1.8, TILE_SIZE * 6.4);
    const tx = clamp(Math.cos(angle) * distance, 0, WORLD_WIDTH);
    const ty = clamp(Math.sin(angle) * distance, 0, WORLD_HEIGHT);
    const seedTile = findBestSpawnTileForHumans(simulationRng);
    const candidateTile = worldToTile(tx, ty);
    const tile = candidateTile && isHumanCompatibleTile(candidateTile) ? candidateTile : seedTile;
    let mergeGenesParents = mergeGenes(human, partner);
    const parentWeights = human.brain.getWeights();
    const mutatedWeights = parentWeights.map(w => {
        const noise = tf.randomNormal(w.shape, 0, 0.05); // mutação pequena
        const mutated = w.add(noise);
        noise.dispose();
        return mutated;
    });
    let newHuman = createHumanFromGenes(tile, simulationRng, 1, Math.random() < 0.5 ? human.sex : partner.sex);
    newHuman.brain = createHumanBrain();
    newHuman.brain.setWeights(mutatedWeights);
    newHuman.genes = mergeGenesParents;
    humans.push(newHuman);
}

function craftItem(human) {
    for (const [itemName, recipe] of Object.entries(CRAFT_RECIPES)) {

        if (human.equipped?.[itemName]) continue;

        const hasWood = countInInventory(human, 'wood') >= recipe.wood;
        const hasStone = countInInventory(human, 'stone') >= recipe.stone;
        const hasMetal = countInInventory(human, 'metal') >= recipe.metal;
        const hasReed = countInInventory(human, 'reed') >= recipe.reed;

        if (hasWood && hasStone && hasMetal && hasReed) {
            removeFromInventory(human, 'wood', recipe.wood);
            removeFromInventory(human, 'stone', recipe.stone);
            removeFromInventory(human, 'metal', recipe.metal);
            removeFromInventory(human, 'reed', recipe.reed);

            if (!human.equipped) human.equipped = {};
            human.equipped[itemName] = true;

            const fx = CRAFT_EFFECTS[itemName];
            human.attackBonus = (human.attackBonus || 0) + fx.attackBonus;
            human.defenseBonus = (human.defenseBonus || 0) + fx.defenseBonus;
            human.visionBonus = (human.visionBonus || 0) + fx.visionBonus;
            human.hasLight = human.hasLight || fx.lightBonus > 0;

            rewardHuman(human, 3.0);
            return;
        }
    }

    rewardHuman(human, -3.0);
}

function mergeGenes(humanA, humanB) {
    const [newColor1, newColor2] = swapLastHexPair(humanA.genes.color, humanB.genes.color);
    const genes = {
        size: humanA.genes.size * 0.2 + humanB.genes.size * 0.2 + randomInt(2, 5),
        speed: humanA.genes.speed * 0.2 + humanB.genes.speed * 0.2 + randomInt(2, 5),
        vision: humanA.genes.speed * 0.2 + humanB.genes.speed * 0.2 + randomInt(2, 5),
        courage: humanA.genes.speed * 0.2 + humanB.genes.speed * 0.2 + randomInt(2, 5),
        efficiency: humanA.genes.speed * 0.2 + humanB.genes.speed * 0.2 + randomInt(2, 5),
        fertility: humanA.genes.speed * 0.2 + humanB.genes.speed * 0.2 + randomInt(2, 5),
        wisdom: humanA.genes.speed * 0.2 + humanB.genes.speed * 0.2 + randomInt(2, 5),
        metabolism: humanA.genes.speed * 0.2 + humanB.genes.speed * 0.2 + randomInt(2, 5),
        dexterity: humanA.genes.speed * 0.2 + humanB.genes.speed * 0.2 + randomInt(2, 5),
        health: randomInt(rng, 80, 80 + (size * 10)),
        color: sumHexColors(newColor1, newColor2)
    }
    return genes;
}

function sumHexColors(color1, color2) {

    const r = Math.min(
        parseInt(color1.slice(1, 3), 16) +
        parseInt(color2.slice(1, 3), 16),
        255
    );

    const g = Math.min(
        parseInt(color1.slice(3, 5), 16) +
        parseInt(color2.slice(3, 5), 16),
        255
    );

    const b = Math.min(
        parseInt(color1.slice(5, 7), 16) +
        parseInt(color2.slice(5, 7), 16),
        255
    );

    return "#" +
        r.toString(16).padStart(2, '0') +
        g.toString(16).padStart(2, '0') +
        b.toString(16).padStart(2, '0');
}

function swapLastHexPair(colorA, colorB) {
    const lastA = colorA.slice(-2);
    const lastB = colorB.slice(-2);

    const newA = colorA.slice(0, 5) + lastB;
    const newB = colorB.slice(0, 5) + lastA;

    return [newA, newB];
}

function attack(human) {

    const target = findNearestHuman(human, TILE_SIZE * 2,
        h => h.id !== human.id
    );

    if (!target) { rewardHuman(human, -10); return; }
    const damage = 20 + (human.attackBonus || 0);
    const blocked = target.defenseBonus || 0;
    target.genes.health -= Math.max(5, damage - blocked * 0.5);

}

function helpOther(human) {

    const other = findNearestHuman(human, TILE_SIZE * 2,
        h => h.id !== human.id && h.alive
    );

    if (!other) { rewardHuman(human, -10); return; }

    if (other.hunger > 70) {
        const foodIndex = human.inventory.findIndex(item =>
            ['berry', 'fish', 'meat', 'mushroom', 'cactus'].includes(item.type)
        );

        if (foodIndex !== -1) {
            const food = human.inventory.splice(foodIndex, 1)[0];
            other.inventory.push(food);
            return;
        }
    }

    if (other.energy < 30) {
        other.energy += 10;
    }
}

function socialize(human) {

    const other = findNearestHuman(human, TILE_SIZE * 2,
        h => h.id !== human.id && h.alive
    );

    if (!other) { rewardHuman(human, -10); return; }
    human.genes.energy = Math.min(100, human.genes.energy + 5);
    other.genes.energy = Math.min(100, other.genes.energy + 5);

}

function flee(human) {

    const visionRadius = TILE_SIZE * 6 + (human.visionBonus || 0);
    const threat = findNearestAnimal(human, visionRadius);
    if (!threat) { rewardHuman(human, -10); return; }
    const dx = human.x - threat.x;
    const dy = human.y - threat.y;

    const length = Math.hypot(dx, dy) || 1;

    const dirX = dx / length;
    const dirY = dy / length;

    const distance = TILE_SIZE * 4;

    human.targetX = human.x + dirX * distance;
    human.targetY = human.y + dirY * distance;
    rewardHuman(human, 1.0);
}

async function saveKnowledgeToStorage() {
    try {
        const allWeights = [];

        for (const human of humans) {
            if (!human.brain) continue;
            const weights = human.brain.getWeights().map(w => ({
                data: Array.from(w.dataSync()),
                shape: w.shape
            }));
            allWeights.push(weights);
        }

        // salva também o melhor cérebro separado
        if (savedGenerationWeights) {
            const best = savedGenerationWeights.map(w => ({
                data: Array.from(w.dataSync()),
                shape: w.shape
            }));
            localStorage.setItem('humanBestBrain', JSON.stringify(best));
        }

        localStorage.setItem('humanCycleCount', String(cycleCount));
        localStorage.setItem('humanKnowledgeSaved', Date.now().toString());

        console.log('Conhecimento salvo.');
    } catch (e) {
        console.warn('Erro ao salvar conhecimento:', e);
    }
}

async function loadKnowledgeFromStorage() {
    try {
        const savedCycle = localStorage.getItem('humanCycleCount');
        if (savedCycle) cycleCount = parseInt(savedCycle) || 0;

        const savedBest = localStorage.getItem('humanBestBrain');
        if (!savedBest) return false;

        const parsed = JSON.parse(savedBest);
        savedGenerationWeights = parsed.map(w =>
            tf.tensor(w.data, w.shape)
        );

        console.log(`Conhecimento carregado. Ciclo #${cycleCount}`);
        return true;
    } catch (e) {
        console.warn('Erro ao carregar conhecimento:', e);
        return false;
    }
}

function equippedIcons(human) {
    if (!human.equipped) return '';
    const icons = { lança: '🗡️', espada: '⚔️', escudo: '🛡️', armadura: '🥋', tocha: '🔦' };
    return Object.keys(human.equipped).map(k => icons[k] || '').join('');
}

function createBase(human) {
    // remover madeira
    let removedWood = 0;
    human.inventory = human.inventory.filter(item => {
        if (item.type === "wood" && removedWood < woodNeeded) {
            removedWood++;
            return false;
        }
        return true;
    });

    // remover pedra
    let removedStone = 0;
    human.inventory = human.inventory.filter(item => {
        if (item.type === "stone" && removedStone < stoneNeeded) {
            removedStone++;
            return false;
        }
        return true;
    });

    const base = {
        id: ++baseIdCounter,
        x: human.x,
        y: human.y,
        health: 100,
        maxHealth: 100,
        size: 18
    };

    bases.push([
        {
            id: 1,
            x: 500,
            y: 300,
            type: "base",
            health: 70,
            maxHealth: 100,
            size: TILE_SIZE * 2
        }
    ]);
}

function repairBase(human) {

    const repairRadius = TILE_SIZE * 3;

    let nearestBase = null;
    let bestDist = repairRadius * repairRadius;

    for (const base of bases) {

        const dist = distanceSq(human.x, human.y, base.x, base.y);

        if (dist < bestDist && base.health < base.maxHealth) {
            bestDist = dist;
            nearestBase = base;
        }
    }

    if (!nearestBase) { rewardHuman(human, -10); return };

    const woodIndex = human.inventory.findIndex(i => i.type === "wood");
    const stoneIndex = human.inventory.findIndex(i => i.type === "stone");

    if (woodIndex === -1 && stoneIndex === -1) { rewardHuman(human, -10); return };

    // consumir recurso
    if (woodIndex !== -1) {
        human.inventory.splice(woodIndex, 1);
        nearestBase.health += 8;
    } else {
        human.inventory.splice(stoneIndex, 1);
        nearestBase.health += 12;
    }

    nearestBase.health = Math.min(nearestBase.health, nearestBase.maxHealth);
}

function trade(human, other) {

    if (!other || !other.alive) return false;

    const distance = Math.hypot(human.x - other.x, human.y - other.y);
    if (distance > 30) return false;

    const humanNeeds = getNeededResource(human);
    const otherNeeds = getNeededResource(other);

    if (!humanNeeds || !otherNeeds) return false;

    const humanGiveIndex = human.inventory.findIndex(i => i.type === otherNeeds);
    const otherGiveIndex = other.inventory.findIndex(i => i.type === humanNeeds);

    if (humanGiveIndex === -1 || otherGiveIndex === -1) return false;

    const humanItem = human.inventory.splice(humanGiveIndex, 1)[0];
    const otherItem = other.inventory.splice(otherGiveIndex, 1)[0];

    human.inventory.push(otherItem);
    other.inventory.push(humanItem);

    return true;
}

function getNeededResource(human) {

    const counts = {
        food: 0,
        wood: 0,
        stone: 0
    };

    for (const item of human.inventory) {
        if (counts[item.type] !== undefined) {
            counts[item.type]++;
        }
    }

    if (counts.food < 2) return "food";
    if (counts.wood < 3) return "wood";
    if (counts.stone < 2) return "stone";

    return null;
}

function drawBases(ctx) {

    for (const base of bases) {
        ctx.fillStyle = "rgba(0,0,0,0.3)";
        ctx.fillRect(base.x - base.size / 2 + 2, base.y - base.size / 2 + 2, base.size, base.size);


        ctx.fillStyle = "#8b5a2b";
        ctx.fillRect(base.x - base.size / 2, base.y - base.size / 2, base.size, base.size);


        ctx.strokeStyle = "#5c3a1e";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(base.x - base.size / 2, base.y - base.size / 2, base.size, base.size);

        ctx.fillStyle = "#a0522d";
        ctx.beginPath();
        ctx.moveTo(base.x - base.size / 2 - 2, base.y - base.size / 2);
        ctx.lineTo(base.x, base.y - base.size / 2 - base.size * 0.5);
        ctx.lineTo(base.x + base.size / 2 + 2, base.y - base.size / 2);
        ctx.closePath();
        ctx.fill();
    }
}

function mulberry32(seed) {
    return function () {
        let t = seed += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

function hash2D(x, y, seed) {
    const value = Math.sin((x * 127.1 + y * 311.7 + seed * 74.7)) * 43758.5453123;
    return value - Math.floor(value);
}

function smoothNoise(x, y, seed) {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const sx = x - x0;
    const sy = y - y0;
    const n00 = hash2D(x0, y0, seed);
    const n10 = hash2D(x1, y0, seed);
    const n01 = hash2D(x0, y1, seed);
    const n11 = hash2D(x1, y1, seed);
    const ix0 = lerp(n00, n10, fade(sx));
    const ix1 = lerp(n01, n11, fade(sx));
    return lerp(ix0, ix1, fade(sy));
}

function fbm(x, y, seed, octaves = 4) {
    let value = 0;
    let amplitude = 0.5;
    let frequency = 1;
    let amplitudeSum = 0;

    for (let i = 0; i < octaves; i++) {
        value += smoothNoise(x * frequency, y * frequency, seed + i * 17) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= 0.5;
        frequency *= 2;
    }

    return value / amplitudeSum;
}

function fade(t) {
    return t * t * (3 - 2 * t);
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function pick(arr, rng) {
    return arr[Math.floor(rng() * arr.length)];
}

function randomBetween(rng, min, max) {
    return min + rng() * (max - min);
}

function randomIntBetween(rng, min, max) {
    return Math.floor(randomBetween(rng, min, max + 1));
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function distanceSq(x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    return dx * dx + dy * dy;
}

function normalize(dx, dy) {
    const length = Math.hypot(dx, dy) || 1;
    return { x: dx / length, y: dy / length };
}

function hexToRgb(hex) {
    const clean = hex.replace('#', '');
    const bigint = parseInt(clean, 16);
    return {
        r: (bigint >> 16) & 255,
        g: (bigint >> 8) & 255,
        b: bigint & 255
    };
}

function shiftHexColor(hex, amount) {
    const { r, g, b } = hexToRgb(hex);
    const nr = clamp(r + amount, 0, 255);
    const ng = clamp(g + amount, 0, 255);
    const nb = clamp(b + amount, 0, 255);
    return `rgb(${nr}, ${ng}, ${nb})`;
}

function drawRoundedRect(x, y, width, height, radius) {
    const r = Math.min(radius, width * 0.5, height * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function drawEntityLabel(x, y, text, options = {}) {
    const bg = options.bg || 'rgba(15, 23, 42, 0.82)';
    const fg = options.fg || '#f8fafc';
    const stroke = options.stroke || 'rgba(255,255,255,0.12)';
    const paddingX = options.paddingX || 5;
    const paddingY = options.paddingY || 3;
    const fontSize = options.fontSize || 10;
    const radius = options.radius || 6;
    const offsetY = options.offsetY || 0;

    ctx.save();
    ctx.font = `600 ${fontSize}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const metrics = ctx.measureText(text);
    const width = metrics.width + paddingX * 2;
    const height = fontSize + paddingY * 2;
    const drawX = x - width * 0.5;
    const drawY = y - height + offsetY;

    drawRoundedRect(drawX, drawY, width, height, radius);
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 0.8;
    ctx.stroke();

    ctx.fillStyle = fg;
    ctx.fillText(text, x, drawY + height * 0.5 + 0.5);
    ctx.restore();
}

function seededDetail(row, col, offset = 0) {
    const value = Math.sin((row * 92821 + col * 68917 + offset * 131) * 0.0001) * 43758.5453;
    return value - Math.floor(value);
}

function getBalancedBiomeThresholds(values, bucketCount) {
    const sorted = [...values].sort((a, b) => a - b);
    const thresholds = [];

    for (let i = 1; i < bucketCount; i++) {
        const index = Math.floor((sorted.length * i) / bucketCount);
        thresholds.push(sorted[Math.min(index, sorted.length - 1)]);
    }

    return thresholds;
}

function determineBalancedBiome(biomeValue, thresholds) {
    const biomeOrder = ['plains', 'forest', 'desert', 'mountain', 'swamp', 'snow', 'lake'];
    for (let i = 0; i < thresholds.length; i++) {
        if (biomeValue <= thresholds[i]) {
            return biomeOrder[i];
        }
    }
    return biomeOrder[biomeOrder.length - 1];
}

function getChunkCoordsFromWorld(x, y) {
    const chunkPixelSize = CHUNK_TILES * TILE_SIZE;
    return {
        chunkCol: Math.floor(x / chunkPixelSize),
        chunkRow: Math.floor(y / chunkPixelSize)
    };
}

function getChunkKey(chunkCol, chunkRow) {
    return `${chunkCol},${chunkRow}`;
}

function cacheTerrainChunk(key, canvas) {
    if (terrainChunks.has(key)) {
        terrainChunks.delete(key);
    }
    terrainChunks.set(key, canvas);
    if (terrainChunks.size > TERRAIN_CACHE_LIMIT) {
        const oldestKey = terrainChunks.keys().next().value;
        if (oldestKey !== undefined) {
            terrainChunks.delete(oldestKey);
        }
    }
}

function buildResourceChunkIndex(resources) {
    resourceChunks.clear();
    for (const resource of resources) {
        const { chunkCol, chunkRow } = getChunkCoordsFromWorld(resource.x, resource.y);
        const key = getChunkKey(chunkCol, chunkRow);
        if (!resourceChunks.has(key)) {
            resourceChunks.set(key, []);
        }
        resourceChunks.get(key).push(resource);
    }
}

function buildAnimalChunkIndex() {
    animalChunks.clear();
    for (const animal of animals) {
        if (!animal.alive) continue;
        const { chunkCol, chunkRow } = getChunkCoordsFromWorld(animal.x, animal.y);
        const key = getChunkKey(chunkCol, chunkRow);
        if (!animalChunks.has(key)) {
            animalChunks.set(key, []);
        }
        animalChunks.get(key).push(animal);
    }
}

function findNearestAnimal(source, radius, filterFn) {
    const radiusSq = radius * radius;
    const { chunkCol, chunkRow } = getChunkCoordsFromWorld(source.x, source.y);
    const chunkRadius = Math.max(1, Math.ceil(radius / (CHUNK_TILES * TILE_SIZE)));
    let nearest = null;
    let nearestDistSq = radiusSq;

    for (let rowOffset = -chunkRadius; rowOffset <= chunkRadius; rowOffset++) {
        for (let colOffset = -chunkRadius; colOffset <= chunkRadius; colOffset++) {
            const key = getChunkKey(chunkCol + colOffset, chunkRow + rowOffset);
            const bucket = animalChunks.get(key);
            if (!bucket) continue;

            for (const candidate of bucket) {
                if (!candidate.alive || candidate.id === source.id) continue;
                if (filterFn && !filterFn(candidate)) continue;
                const distSq = distanceSq(source.x, source.y, candidate.x, candidate.y);
                if (distSq < nearestDistSq) {
                    nearestDistSq = distSq;
                    nearest = candidate;
                }
            }
        }
    }

    return nearest;
}

function buildHumanChunkIndex() {
    humanChunks.clear();
    for (const human of humans) {
        if (!human.alive) continue;
        const { chunkCol, chunkRow } = getChunkCoordsFromWorld(human.x, human.y);
        const key = getChunkKey(chunkCol, chunkRow);
        if (!humanChunks.has(key)) {
            humanChunks.set(key, []);
        }
        humanChunks.get(key).push(human);
    }
}

function findBestSpawnTileForHumans(rng, attempts = 160) {
    let bestTile = null;
    let bestScore = -Infinity;

    for (let i = 0; i < attempts; i++) {

        const tile = findSpawnTileForHumans(rng, 40);
        if (!tile) continue;

        const probe = {
            x: tile.x + TILE_SIZE * 0.5,
            y: tile.y + TILE_SIZE * 0.5
        };

        const water = findNearestResource(probe, TILE_SIZE * 12, new Set(['water']));
        const food = findNearestResource(probe, TILE_SIZE * 12, new Set(['berry', 'mushroom', 'cactus', 'fish']));
        const wood = findNearestResource(probe, TILE_SIZE * 10, new Set(['wood']));

        let score = 0;

        if (water) score += 1000 / (Math.sqrt(distanceSq(probe.x, probe.y, water.x, water.y)) + 1);
        else score -= 20;

        if (food) score += 800 / (Math.sqrt(distanceSq(probe.x, probe.y, food.x, food.y)) + 1);
        else score -= 10;

        if (wood) score += 400 / (Math.sqrt(distanceSq(probe.x, probe.y, wood.x, wood.y)) + 1);

        if (tile.biome === 'forest') score += 6;
        if (tile.biome === 'plains') score += 4;

        if (score > bestScore) {
            bestScore = score;
            bestTile = tile;
        }
    }

    return bestTile;
}

function findSpawnTileForHumans(rng, maxAttempts = 220) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const col = Math.floor(rng() * WORLD_COLS);
        const row = Math.floor(rng() * WORLD_ROWS);
        const tile = world.tiles[row][col];
        if (!tile || tile.hasRiver) continue;
        if (HUMAN_ALLOWED_BIOMES.includes(tile.biome)) return tile;
    }
    return world.tiles[Math.floor(WORLD_ROWS * 0.5)][Math.floor(WORLD_COLS * 0.5)];
}

function isHumanCompatibleTile(tile) {
    return !!tile && HUMAN_ALLOWED_BIOMES.includes(tile.biome) && !tile.hasRiver;
}

function randomHexColor() {
    const hex = Math.floor(Math.random() * 16777215).toString(16);
    return "#" + hex.padStart(6, "0");
}

function createHumanFromGenes(tile, rng, generation = 1, forcedSex = null) {
    const sex = forcedSex || (rng() < 0.5 ? 'M' : 'F');
    let size = randomInt(rng, 4, 12);
    const human = {
        id: ++humanIdCounter,
        name: `Humano ${String(humanIdCounter).padStart(2, '0')}`,
        x: tile.x + TILE_SIZE * 0.5 + (rng() - 0.5) * TILE_SIZE * 0.35,
        y: tile.y + TILE_SIZE * 0.5 + (rng() - 0.5) * TILE_SIZE * 0.35,
        vx: 0,
        vy: 0,
        targetX: tile.x + TILE_SIZE * 0.5,
        targetY: tile.y + TILE_SIZE * 0.5,
        needs: {
            hunger: 50,
            thirst: randomBetween(rng, 24, 32),
            energy: randomBetween(rng, 90, 100)
        },
        ageMs: generation === 1 ? randomBetween(rng, 18000, 56000) : 0,
        generation,
        genes: {
            speed: Number(rng().toFixed(3)),
            vision: Number(clamp((rng() * 0.82) * 0.55, 0.02, 0.98).toFixed(3)),
            courage: Number(clamp((rng() * 0.8) * 0.45, 0.02, 0.98).toFixed(3)),
            efficiency: Number(clamp((rng() * 0.58), 0.02, 0.98).toFixed(3)),
            fertility: Number(clamp((rng() * 0.76), 0.02, 0.98).toFixed(3)),
            wisdom: Number(clamp((rng() * 0.58) * 0.4, 0.02, 0.98).toFixed(3)),
            metabolism: Number(clamp((rng() * 0.82) * 0.55, 0.02, 0.98).toFixed(3)),
            dexterity: Number(clamp((rng() * 0.82) * 0.55, 0.02, 0.98).toFixed(3)),
            size: size,
            health: randomInt(rng, 80, 80 + (size * 10)),
            color: randomHexColor()
        },
        alive: true,
        sex,
        actionLog: [],
        mode: 'wander',
        inventory: [],
        knownActions: [...HUMAN_ACTION_LIBRARY],
        lastX: tile.x + TILE_SIZE * 0.5,
        lastY: tile.y + TILE_SIZE * 0.5,
        brain: createHumanBrain(),
        memory: [],
        mlUpdateCounter: 0,
        decisionCooldownMs: 0,
        lastAction: 'explorando'
    };

    return human;
}

function randomInt(rng, min, max) {
    return Math.floor(rng() * (max - min + 1)) + min;
}

function saveBestBrainWeights() {
    if (!humans.length) return;

    const best = humans.reduce((a, b) =>
        (a.mlUpdateCounter || 0) > (b.mlUpdateCounter || 0) ? a : b
    );

    if (best?.brain) {
        savedGenerationWeights = best.brain.getWeights().map(w => w.clone());
    }
}

function spawnInitialHumans() {
    humans = [];
    bases = [];
    buildSites = [];
    humanIdCounter = 0;
    baseIdCounter = 0;
    buildSiteIdCounter = 0;
    totalBasesBuiltThisRun = 0;
    currentRunActionTotals = {};
    currentRunDeathRecords = [];
    currentRunChildrenBorn = 0;
    currentRunHumansBorn = 0;
    simulationTimeMs = 0;

    const seedTile = findBestSpawnTileForHumans(simulationRng);
    initialHumanFocus = {
        x: seedTile.x + TILE_SIZE * 0.5,
        y: seedTile.y + TILE_SIZE * 0.5
    };

    const initialPopulation = 10;
    const initialSexes = Array.from({ length: initialPopulation }, (_, i) => i % 2 === 0 ? 'M' : 'F');
    for (let i = 0; i < initialPopulation; i++) {
        const angle = (Math.PI * 2 * i) / initialPopulation + randomBetween(simulationRng, -0.16, 0.16);
        const distance = randomBetween(simulationRng, TILE_SIZE * 1.8, TILE_SIZE * 6.4);
        const tx = clamp(initialHumanFocus.x + Math.cos(angle) * distance, 0, WORLD_WIDTH);
        const ty = clamp(initialHumanFocus.y + Math.sin(angle) * distance, 0, WORLD_HEIGHT);
        const candidateTile = worldToTile(tx, ty);
        const tile = candidateTile && isHumanCompatibleTile(candidateTile) ? candidateTile : seedTile;
        humans.push(createHumanFromGenes(tile, simulationRng, 1, initialSexes[i]));
        const created = humans[humans.length - 1];
        created.x = tx;
        created.y = ty;
        created.ageMs = randomBetween(simulationRng, 26000, 68000);
        if (savedGenerationWeights) {
            const mutated = savedGenerationWeights.map(w => {
                const noise = tf.randomNormal(w.shape, 0, 0.05);
                const result = w.add(noise);
                noise.dispose();
                return result;
            });
            created.brain.setWeights(mutated);
        }
    }

    buildHumanChunkIndex();
}

function findNearestResource(source, radius, typeSet) {
    const radiusSq = radius * radius;
    const { chunkCol, chunkRow } = getChunkCoordsFromWorld(source.x, source.y);
    const chunkRadius = Math.max(1, Math.ceil(radius / (CHUNK_TILES * TILE_SIZE)));
    let nearest = null;
    let nearestDistSq = radiusSq;

    for (let rowOffset = -chunkRadius; rowOffset <= chunkRadius; rowOffset++) {
        for (let colOffset = -chunkRadius; colOffset <= chunkRadius; colOffset++) {
            const key = getChunkKey(chunkCol + colOffset, chunkRow + rowOffset);
            const bucket = resourceChunks.get(key);
            if (!bucket) continue;

            for (const resource of bucket) {
                if (resource.amount <= 0) continue;
                if (typeSet && !typeSet.has(resource.type)) continue;
                const distSq = distanceSq(source.x, source.y, resource.x, resource.y);
                if (distSq < nearestDistSq) {
                    nearestDistSq = distSq;
                    nearest = resource;
                }
            }
        }
    }

    return nearest;
}

function buildOverviewLayers() {
    const width = Math.max(1, Math.ceil(WORLD_WIDTH * OVERVIEW_SCALE));
    const height = Math.max(1, Math.ceil(WORLD_HEIGHT * OVERVIEW_SCALE));

    overviewTerrainCanvas = document.createElement('canvas');
    overviewTerrainCanvas.width = width;
    overviewTerrainCanvas.height = height;
    const terrainOverviewCtx = overviewTerrainCanvas.getContext('2d', { alpha: false });
    terrainOverviewCtx.imageSmoothingEnabled = false;

    const scaledTile = TILE_SIZE * OVERVIEW_SCALE;
    for (let row = 0; row < WORLD_ROWS; row++) {
        for (let col = 0; col < WORLD_COLS; col++) {
            const tile = world.tiles[row][col];
            terrainOverviewCtx.fillStyle = BIOMES[tile.biome].color;
            terrainOverviewCtx.fillRect(
                Math.floor(col * TILE_SIZE * OVERVIEW_SCALE),
                Math.floor(row * TILE_SIZE * OVERVIEW_SCALE),
                Math.ceil(scaledTile + 0.5),
                Math.ceil(scaledTile + 0.5)
            );

            if (tile.biome === 'lake' || tile.hasRiver) {
                terrainOverviewCtx.fillStyle = tile.biome === 'lake' ? 'rgba(123, 193, 255, 0.95)' : 'rgba(93, 162, 255, 0.78)';
                terrainOverviewCtx.fillRect(
                    Math.floor(col * TILE_SIZE * OVERVIEW_SCALE),
                    Math.floor((row * TILE_SIZE + TILE_SIZE * 0.3) * OVERVIEW_SCALE),
                    Math.max(1, Math.ceil(scaledTile)),
                    Math.max(1, Math.ceil(TILE_SIZE * 0.4 * OVERVIEW_SCALE))
                );
            }
        }
    }

    overviewResourcesCanvas = document.createElement('canvas');
    overviewResourcesCanvas.width = width;
    overviewResourcesCanvas.height = height;
    const resourceOverviewCtx = overviewResourcesCanvas.getContext('2d');
    resourceOverviewCtx.imageSmoothingEnabled = false;
    for (const resource of world.resources) {
        const style = RESOURCE_STYLE[resource.type];
        resourceOverviewCtx.fillStyle = style.color;
        resourceOverviewCtx.fillRect(Math.floor(resource.x * OVERVIEW_SCALE), Math.floor(resource.y * OVERVIEW_SCALE), 1, 1);
    }
}

function drawOverview() {
    if (!overviewTerrainCanvas) return;
    ctx.drawImage(overviewTerrainCanvas, 0, 0, WORLD_WIDTH, WORLD_HEIGHT);
    if (overviewResourcesCanvas) {
        ctx.globalAlpha = 0.9;
        ctx.drawImage(overviewResourcesCanvas, 0, 0, WORLD_WIDTH, WORLD_HEIGHT);
        ctx.globalAlpha = 1;
    }
}

function requestRender() {
    needsRender = true;
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(renderFrame);
}

function createWorld(seed = Math.floor(Math.random() * 99999999)) {
    const rng = mulberry32(seed);
    const tiles = [];
    const resources = [];
    const biomeCounts = {};
    const tileList = [];
    const biomeValues = [];

    for (const biomeKey of Object.keys(BIOMES)) {
        biomeCounts[biomeKey] = 0;
    }

    for (let row = 0; row < WORLD_ROWS; row++) {
        const rowData = [];
        for (let col = 0; col < WORLD_COLS; col++) {
            const nx = col / WORLD_COLS;
            const ny = row / WORLD_ROWS;

            const macroHeight = fbm(nx * 1.8 + 40, ny * 1.8 + 40, seed + 11, 4);
            const detailHeight = fbm(nx * 6.2 + 160, ny * 6.2 + 160, seed + 19, 4);
            const edgeFalloffX = 1 - Math.abs(nx - 0.5) * 2;
            const edgeFalloffY = 1 - Math.abs(ny - 0.5) * 2;
            const continentality = Math.pow(clamp(Math.min(edgeFalloffX, edgeFalloffY), 0, 1), 0.55);
            const heightValue = clamp((macroHeight * 0.68) + (detailHeight * 0.32) + (continentality * 0.12) - 0.08, 0, 1);

            const macroHumidity = fbm(nx * 2.2 + 300, ny * 2.2 + 300, seed + 23, 3);
            const localHumidity = fbm(nx * 5.1 + 480, ny * 5.1 + 480, seed + 29, 4);
            const humidityValue = clamp((macroHumidity * 0.62) + (localHumidity * 0.38), 0, 1);

            const temperatureBase = 1 - Math.abs(ny - 0.5) * 1.18;
            const temperatureNoise = fbm(nx * 3.5 + 250, ny * 3.5 + 250, seed + 31, 3);
            const elevationCooling = Math.max(0, heightValue - 0.55) * 0.45;
            const temperatureValue = clamp((temperatureBase * 0.72) + (temperatureNoise * 0.28) - elevationCooling, 0, 1);

            const biomeMacro = fbm(nx * 1.15 + 700, ny * 1.15 + 700, seed + 41, 3);
            const biomeDetail = fbm(nx * 2.8 + 920, ny * 2.8 + 920, seed + 53, 2);
            const biomeValue = clamp((biomeMacro * 0.8) + (biomeDetail * 0.2), 0, 1);

            const tile = { col, row, x: col * TILE_SIZE, y: row * TILE_SIZE, biome: 'plains', hasRiver: false, heightValue, humidityValue, temperatureValue, biomeValue };
            rowData.push(tile);
            tileList.push(tile);
            biomeValues.push(biomeValue);
        }
        tiles.push(rowData);
    }

    const thresholds = getBalancedBiomeThresholds(biomeValues, Object.keys(BIOMES).length);
    for (const tile of tileList) {
        tile.biome = determineBalancedBiome(tile.biomeValue, thresholds);
        biomeCounts[tile.biome]++;
        const biomeData = BIOMES[tile.biome];
        const climateBonus = (
            tile.biome === 'forest' ? tile.humidityValue * 0.12 :
                tile.biome === 'desert' ? (1 - tile.humidityValue) * 0.12 :
                    tile.biome === 'snow' ? (1 - tile.temperatureValue) * 0.12 :
                        tile.biome === 'mountain' ? tile.heightValue * 0.12 :
                            tile.biome === 'swamp' ? tile.humidityValue * 0.1 :
                                tile.biome === 'lake' ? 0.08 : 0.06
        );

        const primaryDensity = Math.min(0.58, biomeData.resourceDensity + climateBonus);
        const secondaryDensity = Math.min(0.18, biomeData.resourceDensity * 0.22);

        if (rng() < primaryDensity) resources.push(createResource(tile, biomeData, rng));
        if (rng() < secondaryDensity && tile.biome !== 'lake') resources.push(createResource(tile, biomeData, rng, true));
    }

    carveRivers(tiles, resources, rng);
    buildResourceChunkIndex(resources);
    return { seed, tiles, resources, biomeCounts };
}

function createResource(tile, biomeData, rng, secondary = false) {
    const type = pick(biomeData.resources, rng);
    const offsetScale = secondary ? TILE_SIZE * 0.42 : TILE_SIZE * 0.32;
    return {
        type,
        biome: tile.biome,
        x: tile.x + TILE_SIZE * 0.5 + (rng() - 0.5) * offsetScale,
        y: tile.y + TILE_SIZE * 0.5 + (rng() - 0.5) * offsetScale,
        amount: Math.floor(40 + rng() * 120),
        regrowth: Number((0.2 + rng() * 0.8).toFixed(2))
    };
}

function carveRivers(tiles, resources, rng) {
    const riverCount = 3;
    for (let i = 0; i < riverCount; i++) {
        let col = Math.floor(rng() * WORLD_COLS);
        let row = 0;
        const width = 1;
        for (let step = 0; step < WORLD_ROWS * 1.15; step++) {
            for (let w = -width; w <= width; w++) {
                const cc = clamp(col + w, 0, WORLD_COLS - 1);
                const rr = clamp(row, 0, WORLD_ROWS - 1);
                const tile = tiles[rr][cc];
                tile.hasRiver = true;
                if (rng() < 0.22) {
                    resources.push({
                        type: pick(['fish', 'water'], rng),
                        biome: tile.biome,
                        x: tile.x + TILE_SIZE * rng(),
                        y: tile.y + TILE_SIZE * rng(),
                        amount: Math.floor(50 + rng() * 100),
                        regrowth: Number((0.4 + rng() * 0.6).toFixed(2))
                    });
                }
            }
            row += 1;
            col += Math.floor(rng() * 3) - 1;
            col = clamp(col, 1, WORLD_COLS - 2);
            if (row >= WORLD_ROWS - 1) break;
        }
    }
}

function screenToWorld(screenX, screenY) {
    const width = window.innerWidth;
    const height = window.innerHeight;
    return {
        x: camera.x + (screenX - width / 2) / camera.zoom,
        y: camera.y + (screenY - height / 2) / camera.zoom
    };
}

function worldToTile(worldX, worldY) {
    const col = Math.floor(worldX / TILE_SIZE);
    const row = Math.floor(worldY / TILE_SIZE);
    if (!world || row < 0 || col < 0 || row >= WORLD_ROWS || col >= WORLD_COLS) return null;
    return world.tiles[row][col];
}

function isCompatibleTile(tile, species) {
    if (!tile) return false;
    if (species.biomes.includes(tile.biome)) return true;
    if (tile.hasRiver && (species.biomes.includes('lake') || species.biomes.includes('swamp'))) return true;
    return false;
}

function findSpawnTileForSpecies(species, rng, maxAttempts = 180) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const col = Math.floor(rng() * WORLD_COLS);
        const row = Math.floor(rng() * WORLD_ROWS);
        const tile = world.tiles[row][col];
        if (!isCompatibleTile(tile, species)) continue;
        if (tile.hasRiver && !species.biomes.includes('lake') && !species.biomes.includes('swamp')) continue;
        return tile;
    }
    return null;
}

function createAnimal(speciesId, tile, rng) {
    const species = SPECIES_MAP[speciesId];
    return {
        id: ++animalIdCounter,
        speciesId,
        x: tile.x + TILE_SIZE * 0.5 + (rng() - 0.5) * TILE_SIZE * 0.45,
        y: tile.y + TILE_SIZE * 0.5 + (rng() - 0.5) * TILE_SIZE * 0.45,
        vx: 0,
        vy: 0,
        targetX: tile.x + TILE_SIZE * 0.5,
        targetY: tile.y + TILE_SIZE * 0.5,
        decisionCooldownMs: randomBetween(rng, 500, 1800),
        ageMs: randomBetween(rng, 0, species.lifespanMs * 0.35),
        health: Math.max(22, species.threat * 1.7 + species.size * 3),
        alive: true,
        mode: 'wander',
        colorOffset: Math.round((rng() - 0.5) * 18)
    };
}

function spawnAnimalForSpecies(speciesId, rng = simulationRng) {
    const species = SPECIES_MAP[speciesId];
    const tile = findSpawnTileForSpecies(species, rng);
    if (!tile) return null;
    const animal = createAnimal(speciesId, tile, rng);
    animals.push(animal);
    const state = speciesState.get(speciesId);
    if (state) {
        state.alive += 1;
        state.extinct = false;
    }
    return animal;
}

function createCarcassFromAnimal(animal, now) {
    const species = SPECIES_MAP[animal.speciesId];
    const meatAmount = randomIntBetween(simulationRng, species.meatRange[0], species.meatRange[1]);
    carcasses.push({
        id: ++carcassIdCounter,
        x: animal.x,
        y: animal.y,
        amount: meatAmount,
        speciesId: animal.speciesId,
        createdAt: now,
        decayAt: now + randomBetween(simulationRng, 42000, 78000)
    });
}

function queueRespawnForSpecies(speciesId, now) {
    const species = SPECIES_MAP[speciesId];
    const state = speciesState.get(speciesId);
    if (!state) return;
    if (state.birthsRemaining <= 0) {
        if (state.alive === 0 && state.pendingRespawns === 0) state.extinct = true;
        return;
    }
    if (state.alive + state.pendingRespawns >= species.populationCap) return;

    state.birthsRemaining -= 1;
    state.pendingRespawns += 1;
    respawnQueue.push({
        speciesId,
        spawnAt: now + randomBetween(simulationRng, 9000, 22000)
    });
}

function killAnimal(animal, now) {
    if (!animal || !animal.alive) return;
    animal.alive = false;
    const state = speciesState.get(animal.speciesId);
    if (state) {
        state.alive = Math.max(0, state.alive - 1);
    }
    createCarcassFromAnimal(animal, now);
    queueRespawnForSpecies(animal.speciesId, now);
}

function pickWanderTarget(animal, species, rng = simulationRng) {
    const travel = randomBetween(rng, TILE_SIZE * 1.8, TILE_SIZE * 5.5);
    const angle = randomBetween(rng, 0, Math.PI * 2);
    let targetX = clamp(animal.x + Math.cos(angle) * travel, 0, WORLD_WIDTH);
    let targetY = clamp(animal.y + Math.sin(angle) * travel, 0, WORLD_HEIGHT);
    let targetTile = worldToTile(targetX, targetY);

    if (!isCompatibleTile(targetTile, species)) {
        const spawnTile = findSpawnTileForSpecies(species, rng, 40);
        if (spawnTile) {
            targetX = spawnTile.x + TILE_SIZE * 0.5;
            targetY = spawnTile.y + TILE_SIZE * 0.5;
        }
    }

    animal.targetX = targetX;
    animal.targetY = targetY;
    animal.decisionCooldownMs = randomBetween(rng, 900, 2600);
}

function processRespawns(now) {
    if (!respawnQueue.length) return;
    const remainingQueue = [];
    for (const entry of respawnQueue) {
        if (entry.spawnAt > now) {
            remainingQueue.push(entry);
            continue;
        }
        const state = speciesState.get(entry.speciesId);
        const species = SPECIES_MAP[entry.speciesId];
        if (!state || !species) continue;
        state.pendingRespawns = Math.max(0, state.pendingRespawns - 1);
        if (state.alive >= species.populationCap) continue;
        const spawned = spawnAnimalForSpecies(entry.speciesId, simulationRng);
        if (!spawned && state.birthsRemaining > 0) {
            state.birthsRemaining -= 1;
            state.pendingRespawns += 1;
            remainingQueue.push({ speciesId: entry.speciesId, spawnAt: now + 5000 });
        }
    }
    respawnQueue = remainingQueue;
}

function updateCarcasses(now) {
    carcasses = carcasses.filter((carcass) => carcass.decayAt > now && carcass.amount > 0);
}

function isNearWorldEdge(human, margin = TILE_SIZE * 3) {
    return human.x < margin
        || human.y < margin
        || human.x > WORLD_WIDTH - margin
        || human.y > WORLD_HEIGHT - margin;
}

function updateAnimals(dtMs, now) {

    if (!animals.length) return;

    const dtSeconds = dtMs / 1000;

    buildAnimalChunkIndex();

    for (const animal of animals) {

        if (!animal.alive) continue;

        const species = SPECIES_MAP[animal.speciesId];

        animal.ageMs += dtMs;
        animal.decisionCooldownMs -= dtMs;

        if (animal.health !== undefined && animal.health <= 0) {
            killAnimal(animal, now);
            continue;
        }

        if (animal.ageMs >= species.lifespanMs) {
            killAnimal(animal, now);
            continue;
        }

        let moveVector = null;

        const predator = findNearestAnimal(animal, species.fleeRadius, (other) => {
            const otherSpecies = SPECIES_MAP[other.speciesId];
            return otherSpecies.prey.includes(animal.speciesId) && otherSpecies.threat > species.threat;
        });

        if (predator) {

            animal.mode = 'flee';

            moveVector = normalize(
                animal.x - predator.x,
                animal.y - predator.y
            );

        } else {

            let prey = null;

            if (species.prey.length) {
                prey = findNearestAnimal(
                    animal,
                    species.vision,
                    (other) => species.prey.includes(other.speciesId)
                );
            }

            let humanPrey = null;

            if (!prey && (species.diet === "carnivore" || species.diet === "omnivore")) {
                humanPrey = findNearestHuman(animal, species.vision);
            }

            if (prey) {

                const distSq = distanceSq(animal.x, animal.y, prey.x, prey.y);

                animal.mode = 'hunt';

                moveVector = normalize(
                    prey.x - animal.x,
                    prey.y - animal.y
                );

                if (distSq <= species.attackRange * species.attackRange) {
                    killAnimal(prey, now);
                    animal.decisionCooldownMs = 250;
                }

            }
            else if (humanPrey) {

                const distSq = distanceSq(animal.x, animal.y, humanPrey.x, humanPrey.y);

                animal.mode = 'huntHuman';

                moveVector = normalize(
                    humanPrey.x - animal.x,
                    humanPrey.y - animal.y
                );

                if (distSq <= species.attackRange * species.attackRange) {
                    const defense = humanPrey.defenseBonus || 0;
                    const damage = Math.max(1, species.threat * 0.4 - defense * 0.3);
                    humanPrey.genes.health -= damage;

                    if (humanPrey.genes.health <= 0) {
                        humanPrey.alive = false;
                    }

                    animal.decisionCooldownMs = 250;
                }

            }
            else {

                if (
                    animal.decisionCooldownMs <= 0 ||
                    distanceSq(animal.x, animal.y, animal.targetX, animal.targetY) < 18 * 18
                ) {
                    pickWanderTarget(animal, species);
                }

                animal.mode = 'wander';

                moveVector = normalize(
                    animal.targetX - animal.x,
                    animal.targetY - animal.y
                );
            }
        }

        const speedMultiplier =
            animal.mode === 'flee'
                ? 1.15
                : animal.mode === 'hunt' || animal.mode === 'huntHuman'
                    ? 1.05
                    : 0.72;

        const speed = species.speed * speedMultiplier;

        animal.vx = moveVector.x * speed;
        animal.vy = moveVector.y * speed;

        const nextX = clamp(animal.x + animal.vx * dtSeconds, 0, WORLD_WIDTH);
        const nextY = clamp(animal.y + animal.vy * dtSeconds, 0, WORLD_HEIGHT);

        const nextTile = worldToTile(nextX, nextY);

        if (isCompatibleTile(nextTile, species)) {
            animal.x = nextX;
            animal.y = nextY;
        } else {
            animal.decisionCooldownMs = 0;
        }
    }

    animals = animals.filter((animal) => animal.alive);

    buildAnimalChunkIndex();
}

function initializeFauna() {
    animals = [];
    carcasses = [];
    respawnQueue = [];
    speciesState = new Map();
    animalIdCounter = 0;
    carcassIdCounter = 0;
    simulationRng = mulberry32(world.seed + 90210);

    for (const species of ANIMAL_SPECIES) {
        speciesState.set(species.id, {
            alive: 0,
            pendingRespawns: 0,
            birthsRemaining: species.respawnBudget,
            extinct: false
        });
    }

    for (const species of ANIMAL_SPECIES) {
        for (let i = 0; i < species.initialPopulation; i++) {
            spawnAnimalForSpecies(species.id, simulationRng);
        }
    }

    buildAnimalChunkIndex();
    buildFaunaLegend();
}

function formatDietLabel(diet) {
    return diet === 'carnivore' ? 'Carnívoro' : diet === 'omnivore' ? 'Onívoro' : 'Herbívoro';
}

function createTerrainChunk(chunkCol, chunkRow) {
    const chunkCanvas = document.createElement('canvas');
    const chunkWidthTiles = Math.min(CHUNK_TILES, WORLD_COLS - chunkCol * CHUNK_TILES);
    const chunkHeightTiles = Math.min(CHUNK_TILES, WORLD_ROWS - chunkRow * CHUNK_TILES);
    const chunkPixelWidth = chunkWidthTiles * TILE_SIZE;
    const chunkPixelHeight = chunkHeightTiles * TILE_SIZE;

    chunkCanvas.width = chunkPixelWidth;
    chunkCanvas.height = chunkPixelHeight;
    const chunkCtx = chunkCanvas.getContext('2d', { alpha: false });
    chunkCtx.imageSmoothingEnabled = false;

    for (let localRow = 0; localRow < chunkHeightTiles; localRow++) {
        for (let localCol = 0; localCol < chunkWidthTiles; localCol++) {
            const worldRow = chunkRow * CHUNK_TILES + localRow;
            const worldCol = chunkCol * CHUNK_TILES + localCol;
            const tile = world.tiles[worldRow][worldCol];
            const baseColor = BIOMES[tile.biome].color;
            const detailA = seededDetail(worldRow, worldCol, 1);
            const detailB = seededDetail(worldRow, worldCol, 2);
            const drawX = localCol * TILE_SIZE;
            const drawY = localRow * TILE_SIZE;

            chunkCtx.fillStyle = shiftHexColor(baseColor, Math.round((detailA - 0.5) * 20));
            chunkCtx.fillRect(drawX, drawY, TILE_SIZE + 1, TILE_SIZE + 1);

            if (tile.biome === 'lake' || tile.hasRiver) {
                chunkCtx.fillStyle = tile.biome === 'lake' ? 'rgba(123, 193, 255, 0.92)' : 'rgba(93, 162, 255, 0.7)';
                chunkCtx.beginPath();
                chunkCtx.ellipse(
                    drawX + TILE_SIZE * 0.5,
                    drawY + TILE_SIZE * (0.52 + detailA * 0.04),
                    TILE_SIZE * (0.48 + detailA * 0.08),
                    TILE_SIZE * (0.22 + detailB * 0.04),
                    detailB * 0.35,
                    0,
                    Math.PI * 2
                );
                chunkCtx.fill();
            } else {
                chunkCtx.fillStyle = `rgba(255,255,255,${0.015 + detailA * 0.015})`;
                chunkCtx.fillRect(
                    drawX + TILE_SIZE * (0.18 + detailA * 0.26),
                    drawY + TILE_SIZE * (0.18 + detailB * 0.26),
                    2,
                    2
                );
            }
        }
    }

    cacheTerrainChunk(getChunkKey(chunkCol, chunkRow), chunkCanvas);
    return chunkCanvas;
}

function getTerrainChunk(chunkCol, chunkRow) {
    const key = getChunkKey(chunkCol, chunkRow);
    if (terrainChunks.has(key)) {
        const cached = terrainChunks.get(key);
        terrainChunks.delete(key);
        terrainChunks.set(key, cached);
        return cached;
    }
    return createTerrainChunk(chunkCol, chunkRow);
}

function renderVisibleTerrain(viewLeft, viewTop, viewRight, viewBottom) {
    const chunkPixelSize = CHUNK_TILES * TILE_SIZE;
    const maxChunkCol = Math.ceil(WORLD_COLS / CHUNK_TILES) - 1;
    const maxChunkRow = Math.ceil(WORLD_ROWS / CHUNK_TILES) - 1;
    const startChunkCol = clamp(Math.floor(viewLeft / chunkPixelSize) - 1, 0, maxChunkCol);
    const endChunkCol = clamp(Math.floor(viewRight / chunkPixelSize) + 1, 0, maxChunkCol);
    const startChunkRow = clamp(Math.floor(viewTop / chunkPixelSize) - 1, 0, maxChunkRow);
    const endChunkRow = clamp(Math.floor(viewBottom / chunkPixelSize) + 1, 0, maxChunkRow);

    for (let chunkRow = startChunkRow; chunkRow <= endChunkRow; chunkRow++) {
        for (let chunkCol = startChunkCol; chunkCol <= endChunkCol; chunkCol++) {
            const chunkCanvas = getTerrainChunk(chunkCol, chunkRow);
            ctx.drawImage(chunkCanvas, chunkCol * chunkPixelSize, chunkRow * chunkPixelSize);
        }
    }
}

function drawVisibleResources(viewLeft, viewTop, viewRight, viewBottom) {
    const chunkPixelSize = CHUNK_TILES * TILE_SIZE;
    const maxChunkCol = Math.ceil(WORLD_COLS / CHUNK_TILES) - 1;
    const maxChunkRow = Math.ceil(WORLD_ROWS / CHUNK_TILES) - 1;
    const startChunkCol = clamp(Math.floor(viewLeft / chunkPixelSize) - 1, 0, maxChunkCol);
    const endChunkCol = clamp(Math.floor(viewRight / chunkPixelSize) + 1, 0, maxChunkCol);
    const startChunkRow = clamp(Math.floor(viewTop / chunkPixelSize) - 1, 0, maxChunkRow);
    const endChunkRow = clamp(Math.floor(viewBottom / chunkPixelSize) + 1, 0, maxChunkRow);

    for (let chunkRow = startChunkRow; chunkRow <= endChunkRow; chunkRow++) {
        for (let chunkCol = startChunkCol; chunkCol <= endChunkCol; chunkCol++) {
            const chunkResources = resourceChunks.get(getChunkKey(chunkCol, chunkRow));
            if (!chunkResources) continue;
            for (const resource of chunkResources) {
                if (resource.x < viewLeft - 20 || resource.x > viewRight + 20 || resource.y < viewTop - 20 || resource.y > viewBottom + 20) continue;
                if (resource.amount <= 0) continue;
                const style = RESOURCE_STYLE[resource.type];
                ctx.beginPath();
                ctx.fillStyle = style.color;
                ctx.arc(resource.x, resource.y, style.radius, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }
}

function drawVisibleCarcasses(viewLeft, viewTop, viewRight, viewBottom) {
    for (const carcass of carcasses) {
        if (carcass.x < viewLeft - 24 || carcass.x > viewRight + 24 || carcass.y < viewTop - 24 || carcass.y > viewBottom + 24) continue;
        const radius = clamp(3 + carcass.amount * 0.06, 3, 9);
        ctx.beginPath();
        ctx.fillStyle = 'rgba(120, 20, 20, 0.88)';
        ctx.arc(carcass.x, carcass.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 219, 112, 0.35)';
        ctx.lineWidth = 1.25;
        ctx.arc(carcass.x, carcass.y, radius + 1.8, 0, Math.PI * 2);
        ctx.stroke();
    }
}

function drawVisibleAnimals(viewLeft, viewTop, viewRight, viewBottom) {
    for (const animal of animals) {
        if (!animal.alive) continue;
        if (animal.x < viewLeft - 30 || animal.x > viewRight + 30 || animal.y < viewTop - 30 || animal.y > viewBottom + 30) continue;
        const species = SPECIES_MAP[animal.speciesId];
        const dir = normalize(animal.vx, animal.vy);
        const radius = species.size;
        const color = shiftHexColor(species.color, animal.colorOffset);

        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(animal.x, animal.y, radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = species.diet === 'carnivore' ? 'rgba(255,244,244,0.92)' : 'rgba(255,255,255,0.78)';
        ctx.arc(animal.x + dir.x * (radius * 0.55), animal.y + dir.y * (radius * 0.55), Math.max(1.6, radius * 0.28), 0, Math.PI * 2);
        ctx.fill();

        if (species.diet === 'carnivore' || species.threat >= 50) {
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(255, 102, 102, 0.5)';
            ctx.lineWidth = 1;
            ctx.arc(animal.x, animal.y, radius + 1.8, 0, Math.PI * 2);
            ctx.stroke();
        }

        if (camera.zoom >= ENTITY_LABEL_ZOOM_THRESHOLD) {
            drawEntityLabel(animal.x, animal.y - radius - 6, species.name, {
                bg: species.diet === 'carnivore' ? 'rgba(80, 16, 16, 0.84)' : 'rgba(15, 23, 42, 0.82)',
                fg: '#f8fafc',
                stroke: species.diet === 'carnivore' ? 'rgba(255,140,140,0.35)' : 'rgba(255,255,255,0.12)',
                fontSize: 9,
                radius: 5
            });
        }
    }
}

function drawVisibleBuildSites(viewLeft, viewTop, viewRight, viewBottom) {
    for (const site of buildSites) {
        if (site.x < viewLeft - 50 || site.x > viewRight + 50 || site.y < viewTop - 50 || site.y > viewBottom + 50) continue;
        const width = 18;
        const height = 12;
        ctx.beginPath();
        ctx.fillStyle = 'rgba(120, 113, 108, 0.82)';
        ctx.fillRect(site.x - width * 0.5, site.y - height * 0.5, width, height);
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.9)';
        ctx.lineWidth = 1.4;
        ctx.strokeRect(site.x - width * 0.5, site.y - height * 0.5, width, height);
        const progress = ((site.stored.wood + site.stored.stone) / (HUMAN_BASE_BUILD_COST.wood + HUMAN_BASE_BUILD_COST.stone));
        ctx.fillStyle = 'rgba(251, 191, 36, 0.85)';
        ctx.fillRect(site.x - 10, site.y + 10, 20 * progress, 3);
        if (camera.zoom >= ENTITY_LABEL_ZOOM_THRESHOLD) {
            drawEntityLabel(site.x, site.y - 16, `${site.name} • M ${Math.round(site.stored.wood)}/${HUMAN_BASE_BUILD_COST.wood} • P ${Math.round(site.stored.stone)}/${HUMAN_BASE_BUILD_COST.stone}`, {
                bg: 'rgba(120, 53, 15, 0.88)',
                fg: '#fef3c7',
                stroke: 'rgba(253,224,71,0.35)',
                fontSize: 9,
                radius: 6
            });
        }
    }
}

function drawVisibleBases(viewLeft, viewTop, viewRight, viewBottom) {
    for (const base of bases) {
        if (base.x < viewLeft - 60 || base.x > viewRight + 60 || base.y < viewTop - 60 || base.y > viewBottom + 60) continue;
        const size = 14;
        ctx.beginPath();
        ctx.fillStyle = 'rgba(110, 74, 39, 0.92)';
        ctx.moveTo(base.x - size, base.y + size * 0.55);
        ctx.lineTo(base.x + size, base.y + size * 0.55);
        ctx.lineTo(base.x + size, base.y - size * 0.25);
        ctx.lineTo(base.x - size, base.y - size * 0.25);
        ctx.closePath();
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = 'rgba(120, 53, 15, 0.95)';
        ctx.moveTo(base.x - size * 1.15, base.y - size * 0.2);
        ctx.lineTo(base.x, base.y - size * 1.05);
        ctx.lineTo(base.x + size * 1.15, base.y - size * 0.2);
        ctx.closePath();
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = 'rgba(253, 224, 71, 0.28)';
        ctx.arc(base.x, base.y, base.radius + 6, 0, Math.PI * 2);
        ctx.fill();

        if (camera.zoom >= ENTITY_LABEL_ZOOM_THRESHOLD) {
            drawEntityLabel(base.x, base.y - size - 10, `${base.name} • ${Math.round(base.integrity)}%`, {
                bg: 'rgba(67, 56, 202, 0.86)',
                fg: '#eef2ff',
                stroke: 'rgba(199,210,254,0.35)',
                fontSize: 9,
                radius: 6
            });
        }
    }
}

function drawVisibleHumans(viewLeft, viewTop, viewRight, viewBottom) {
    for (const human of humans) {
        if (!human.alive) continue;
        if (human.x < viewLeft - 36 || human.x > viewRight + 36 || human.y < viewTop - 36 || human.y > viewBottom + 36) continue;
        const dir = normalize(human.vx, human.vy);
        const isFemale = human.sex === 'F';
        const bodyColor = human.genes.color;
        const capeColor = isFemale ? '#ec4899' : '#2563eb';
        const torsoRadius = human.genes.size + human.generation * 0.12;

        ctx.beginPath();
        ctx.fillStyle = isFemale ? '#ec4899' : '#2563eb';
        ctx.arc(human.x, human.y + 1.8, torsoRadius + 3.2, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = capeColor;
        ctx.arc(human.x, human.y + 1.8, torsoRadius + 2.2, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = bodyColor;
        ctx.arc(human.x, human.y + 2.4, torsoRadius, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = 'rgba(255,236,210,0.98)';
        ctx.arc(human.x + dir.x * 1.8, human.y - 5 + dir.y * 1.6, 3.4, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.strokeStyle = 'rgba(8, 47, 73, 0.95)';
        ctx.lineWidth = 1.1;
        ctx.arc(human.x + dir.x * 1.8, human.y - 5 + dir.y * 1.6, 3.4, 0, Math.PI * 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255,255,255,0.95)';
        ctx.lineWidth = 1.4;
        ctx.arc(human.x, human.y + 1.8, torsoRadius + 3.4, 0, Math.PI * 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255,255,255,0.85)';
        ctx.lineWidth = 1;
        ctx.moveTo(human.x + dir.x * (torsoRadius + 1), human.y + dir.y * (torsoRadius + 1));
        ctx.lineTo(human.x + dir.x * (torsoRadius + 7), human.y + dir.y * (torsoRadius + 7));
        ctx.stroke();

        if (camera.zoom >= ENTITY_LABEL_ZOOM_THRESHOLD) {
            drawEntityLabel(human.x, human.y - torsoRadius - 10, `${human.name} • ${human.lastAction || 'explorando'} • ❤️${Math.round(human.genes.health)}`, {
                bg: 'rgba(15, 23, 42, 0.9)',
                fg: '#dbeafe',
                stroke: 'rgba(147,197,253,0.45)',
                fontSize: 10,
                radius: 6
            });
        }
    }
}

function renderFrame(now = performance.now()) {
    renderQueued = false;
    if (!needsRender || !world) return;
    needsRender = false;

    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    ctx.save();
    ctx.translate(window.innerWidth / 2, window.innerHeight / 2);
    ctx.scale(camera.zoom, camera.zoom);
    ctx.translate(-camera.x, -camera.y);

    const viewLeft = camera.x - window.innerWidth / 2 / camera.zoom;
    const viewTop = camera.y - window.innerHeight / 2 / camera.zoom;
    const viewRight = camera.x + window.innerWidth / 2 / camera.zoom;
    const viewBottom = camera.y + window.innerHeight / 2 / camera.zoom;

    if (camera.zoom <= OVERVIEW_ZOOM_THRESHOLD) {
        drawOverview();
    } else {
        renderVisibleTerrain(viewLeft, viewTop, viewRight, viewBottom);
        if (camera.zoom >= RESOURCE_DRAW_ZOOM_THRESHOLD) drawVisibleResources(viewLeft, viewTop, viewRight, viewBottom);
        if (camera.zoom >= CARCASS_DRAW_ZOOM_THRESHOLD) drawVisibleCarcasses(viewLeft, viewTop, viewRight, viewBottom);
        if (camera.zoom >= ANIMAL_DRAW_ZOOM_THRESHOLD) drawVisibleAnimals(viewLeft, viewTop, viewRight, viewBottom);
        if (camera.zoom >= HUMAN_DRAW_ZOOM_THRESHOLD) drawVisibleHumans(viewLeft, viewTop, viewRight, viewBottom);
        drawBases(ctx);
    }

    ctx.restore();
    if (now - lastHudUpdate >= HUD_UPDATE_INTERVAL_MS) {
        updateHud();
        lastHudUpdate = now;
    }
}

function updateHud() {
    if (worldSizeLabel) worldSizeLabel.textContent = `${WORLD_COLS} x ${WORLD_ROWS} tiles`;
    if (resourceCountLabel && world) resourceCountLabel.textContent = world.resources.filter((resource) => resource.amount > 0).length.toLocaleString('pt-BR');
    const tile = worldToTile(camera.x, camera.y);
    if (focusBiomeLabel) focusBiomeLabel.textContent = tile ? BIOMES[tile.biome].name : '-';
    if (cameraInfoLabel) cameraInfoLabel.textContent = `${camera.zoom.toFixed(2)}x • ${Math.round(camera.x)}, ${Math.round(camera.y)}`;

    if (animalCountLabel) animalCountLabel.textContent = animals.length.toLocaleString('pt-BR');
    if (carcassCountLabel) carcassCountLabel.textContent = carcasses.length.toLocaleString('pt-BR');
    if (birthBudgetLabel) {
        const totalBirthsRemaining = Array.from(speciesState.values()).reduce((sum, state) => sum + state.birthsRemaining, 0);
        birthBudgetLabel.textContent = totalBirthsRemaining.toLocaleString('pt-BR');
    }
    if (extinctSpeciesLabel) {
        const extinctCount = Array.from(speciesState.values()).filter((state) => state.extinct).length;
        extinctSpeciesLabel.textContent = extinctCount.toLocaleString('pt-BR');
    }

    if (humanCountLabel) humanCountLabel.textContent = humans.length.toLocaleString('pt-BR');
    if (humanGenerationLabel) {
        humanGenerationLabel.textContent = '∞';
    }
    if (humanWisdomLabel) {
        const avgWisdom = humans.length
            ? humans.reduce((sum, h) => {
                const updates = h.mlUpdateCounter || 0;
                return sum + Math.min(1, updates / 100);
            }, 0) / humans.length
            : 0;
        humanWisdomLabel.textContent = `${Math.round(avgWisdom * 100)}%`;
    }
    if (humanEfficiencyLabel) {
        const avgEfficiency = humans.length
            ? humans.reduce((sum, h) => {
                const mem = h.memory || [];
                if (!mem.length) return sum;
                const positive = mem.filter(e => e.reward > 0).length;
                return sum + (positive / mem.length);
            }, 0) / humans.length
            : 0;
        humanEfficiencyLabel.textContent = `${Math.round(avgEfficiency * 100)}%`;
    }
    if (humanBaseLabel) humanBaseLabel.textContent = bases.length.toLocaleString('pt-BR');
    if (humanTopScoreLabel) {
        const topScore = humans.length ? Math.max(...humans.map((human) => human.score || 0)) : 0;
        humanTopScoreLabel.textContent = topScore.toLocaleString('pt-BR');
    }

    buildFaunaLegend();
    if (performance.now() - lastHumanLegendUpdate >= HUMAN_LEGEND_UPDATE_INTERVAL_MS) {
        buildHumanLegend();
        lastHumanLegendUpdate = performance.now();
    }
}

function buildHumanLegend() {
    if (!humanLegend) return;

    const sortedHumans = [...humans].sort((a, b) =>
        (b.score || 0) - (a.score || 0) ||
        b.generation - a.generation ||
        b.health - a.health ||
        a.id - b.id
    );

    humanLegend.innerHTML = sortedHumans.map((human) => {
        const currentAction = human.currentTensorflowAction || human.lastAction || 'explorando';

        const adaptationPercent = Math.round(
            clamp((human.mlUpdateCounter || 0) / 80, 0, 1) *
            (human.genes?.wisdom || 1) * 100
        );

        return `
        <div class="species-entry human-focus-target" data-human-id="${human.id}">
            <div class="species-main">
            <span class="swatch" style="background:${shiftHexColor('#f4c38b', human.colorOffset)}"></span>
            <div class="species-info">
                <div class="species-name">${human.name} ${equippedIcons(human)} • ${human.lastAction || 'explorando'} • ❤️${Math.round(human.genes.health)}</div>
                <div class="species-sub">Energia ${Math.round(human.needs.energy)} • Idade ${Math.floor(human.ageMs / 1000 * 0.21)} anos • Posição ${Math.round(human.x)}, ${Math.round(human.y)}</div>
                <div class="species-sub">Adaptação: ${adaptationPercent}%</div>
            </div>
            </div>
        </div>
        `;
    }).join('');
}

function buildFaunaLegend() {
    if (!faunaLegend) return;
    faunaLegend.innerHTML = ANIMAL_SPECIES.map((species) => {
        const state = speciesState.get(species.id) || { alive: 0, birthsRemaining: species.respawnBudget, extinct: false };
        const biomes = species.biomes.map((biomeId) => BIOMES[biomeId]?.name || biomeId).join(', ');
        return `
          <div class="species-entry">
            <div class="species-main">
              <span class="swatch" style="background:${species.color}"></span>
              <div class="species-info">
                <div class="species-name">${species.name}</div>
              </div>
            </div>
            <div class="species-meta">
              <span class="species-chip${state.extinct ? ' extinct' : ''}">${state.extinct ? 'Extinta' : `${state.alive}/${species.populationCap} vivos`}</span>
              <span class="species-chip">Nasc.: ${state.birthsRemaining}</span>
            </div>
          </div>
        `;
    }).join('');
}

async function regenerateWorld() {
    terrainChunks.clear();
    overviewTerrainCanvas = null;
    overviewResourcesCanvas = null;
    currentCycleFinalized = false;
    currentCycleSummary = null;
    extinctionModalActive = false;
    simRunning = true;
    world = createWorld();
    buildOverviewLayers();
    initializeFauna();
    spawnInitialHumans();
    camera.x = initialHumanFocus ? initialHumanFocus.x : WORLD_WIDTH * 0.5;
    camera.y = initialHumanFocus ? initialHumanFocus.y : WORLD_HEIGHT * 0.5;
    camera.zoom = 0.9;
    updateHud();
    requestRender();
}

function removeResource(resource) {
    if (!resource) return;

    resource.amount -= 1;
    if (resource.amount <= 0) {
        const { chunkCol, chunkRow } = getChunkCoordsFromWorld(resource.x, resource.y);
        const key = getChunkKey(chunkCol, chunkRow);
        const bucket = resourceChunks.get(key);
        if (bucket) {
            const idx = bucket.indexOf(resource);
            if (idx !== -1) bucket.splice(idx, 1);
        }
    }
}

async function simulationStep() {
    if (!world || !simRunning) return;
    const now = performance.now();
    simulationTimeMs += SIM_TICK_MS;
    updateAnimals(SIM_TICK_MS, now);
    processRespawns(now);
    updateCarcasses(now);
    chooseAllHumanActions();
    updateHumans(SIM_TICK_MS, now);
    for (const [speciesId, state] of speciesState.entries()) {
        if (state.alive === 0 && state.pendingRespawns === 0 && state.birthsRemaining <= 0) {
            state.extinct = true;
        } else if (state.alive > 0) {
            state.extinct = false;
        }
    }

    requestRender();
    if (humans.length === 0 && !extinctionModalActive) {
        extinctionModalActive = true;
        simRunning = false;
        saveBestBrainWeights();
        cycleCount++;
        showExtinctionModal();
    }
}

function startSimulationLoop() {
    if (simulationLoopId) clearInterval(simulationLoopId);
    simulationLoopId = setInterval(simulationStep, SIM_TICK_MS);
}

function getTouchDistance(touches) {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.hypot(dx, dy);
}

function getTouchCenter(touches) {
    return {
        x: (touches[0].clientX + touches[1].clientX) / 2,
        y: (touches[0].clientY + touches[1].clientY) / 2
    };
}

function EventsListener() {
    setInterval(() => {
        if (simRunning) saveKnowledgeToStorage();
    }, 30000);


    window.addEventListener('beforeunload', () => {
        saveBestBrainWeights();
        saveKnowledgeToStorage();
    });

    document.querySelectorAll('.collapse-panel > summary').forEach((summary) => {
        summary.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            const panel = summary.parentElement;
            panel.toggleAttribute('open');
        });
    });

    zoomInButton?.addEventListener('click', () => {
        zoomCameraAt(window.innerWidth * 0.5, window.innerHeight * 0.5, 1.18);
    });

    zoomOutButton?.addEventListener('click', () => {
        zoomCameraAt(window.innerWidth * 0.5, window.innerHeight * 0.5, 0.84);
    });

    pauseButton?.addEventListener('click', () => {
        simRunning = !simRunning;
        if (extinctionModalActive && simRunning) simRunning = false;
    });

    canvas.addEventListener('mousedown', (event) => {
        drag.active = true;
        drag.startX = event.clientX;
        drag.startY = event.clientY;
        drag.cameraStartX = camera.x;
        drag.cameraStartY = camera.y;
        canvas.classList.add('dragging');
    });

    window.addEventListener('mouseup', () => {
        drag.active = false;
        canvas.classList.remove('dragging');
    });

    window.addEventListener('mousemove', (event) => {
        if (!drag.active) return;
        const dx = (event.clientX - drag.startX) / camera.zoom;
        const dy = (event.clientY - drag.startY) / camera.zoom;
        camera.x = clamp(drag.cameraStartX - dx, 0, WORLD_WIDTH);
        camera.y = clamp(drag.cameraStartY - dy, 0, WORLD_HEIGHT);
        requestRender();
    });

    canvas.addEventListener('touchstart', (event) => {
        if (event.touches.length === 1) {
            const touch = event.touches[0];
            touchState.active = true;
            touchState.mode = 'pan';
            touchState.lastX = touch.clientX;
            touchState.lastY = touch.clientY;
            touchState.cameraStartX = camera.x;
            touchState.cameraStartY = camera.y;
        } else if (event.touches.length >= 2) {
            const center = getTouchCenter(event.touches);
            touchState.active = true;
            touchState.mode = 'pinch';
            touchState.startDistance = getTouchDistance(event.touches);
            touchState.startZoom = camera.zoom;
            touchState.startCenterX = center.x;
            touchState.startCenterY = center.y;
            touchState.cameraStartX = camera.x;
            touchState.cameraStartY = camera.y;
        }
    }, { passive: true });

    canvas.addEventListener('touchmove', (event) => {
        if (!touchState.active) return;
        if (touchState.mode === 'pan' && event.touches.length === 1) {
            event.preventDefault();
            const touch = event.touches[0];
            const dx = (touch.clientX - touchState.lastX) / camera.zoom;
            const dy = (touch.clientY - touchState.lastY) / camera.zoom;
            camera.x = clamp(camera.x - dx, 0, WORLD_WIDTH);
            camera.y = clamp(camera.y - dy, 0, WORLD_HEIGHT);
            touchState.lastX = touch.clientX;
            touchState.lastY = touch.clientY;
            requestRender();
        } else if (event.touches.length >= 2) {
            event.preventDefault();
            const center = getTouchCenter(event.touches);
            const distance = Math.max(12, getTouchDistance(event.touches));
            const scale = distance / Math.max(12, touchState.startDistance || distance);
            const before = screenToWorld(center.x, center.y);
            camera.zoom = clamp(touchState.startZoom * scale, 0.08, 2.4);
            const after = screenToWorld(center.x, center.y);
            camera.x = clamp(camera.x + (before.x - after.x), 0, WORLD_WIDTH);
            camera.y = clamp(camera.y + (before.y - after.y), 0, WORLD_HEIGHT);
            requestRender();
        }
    }, { passive: false });

    canvas.addEventListener('touchend', (event) => {
        if (event.touches.length === 0) {
            touchState.active = false;
            touchState.mode = null;
        } else if (event.touches.length === 1) {
            const touch = event.touches[0];
            touchState.active = true;
            touchState.mode = 'pan';
            touchState.lastX = touch.clientX;
            touchState.lastY = touch.clientY;
        }
    });

    canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const mouseBefore = screenToWorld(event.clientX, event.clientY);
        const zoomIntensity = event.deltaY < 0 ? 1.08 : 0.94;
        const zoomFactor = camera.zoom <= OVERVIEW_ZOOM_THRESHOLD ? (event.deltaY < 0 ? 1.05 : 0.97) : zoomIntensity;
        camera.zoom = clamp(camera.zoom * zoomFactor, 0.08, 2.4);
        const mouseAfter = screenToWorld(event.clientX, event.clientY);
        camera.x += mouseBefore.x - mouseAfter.x;
        camera.y += mouseBefore.y - mouseAfter.y;
        camera.x = clamp(camera.x, 0, WORLD_WIDTH);
        camera.y = clamp(camera.y, 0, WORLD_HEIGHT);
        requestRender();
    }, { passive: false });

    window.addEventListener('resize', () => { resizeCanvas(); updateMobileUIState(); requestRender(); });
}

function showExtinctionModal() {
    document.getElementById('extinction-modal')?.remove();
    const lastSaved = localStorage.getItem('humanKnowledgeSaved');
    const lastSavedText = lastSaved
        ? `Último save: ${new Date(parseInt(lastSaved)).toLocaleString('pt-BR')}`
        : '';
    const modal = document.createElement('div');
    modal.id = 'extinction-modal';
    modal.style.cssText = `
        position: fixed; inset: 0; background: rgba(0,0,0,0.75);
        display: flex; align-items: center; justify-content: center;
        z-index: 9999; font-family: Arial, sans-serif;
    `;

    modal.innerHTML = `
        <div style="
            background: #1a1a2e; border: 1px solid #444; border-radius: 16px;
            padding: 40px 48px; max-width: 480px; text-align: center; color: #f0f0f0;
        ">
            <div style="font-size: 48px; margin-bottom: 12px;">💀</div>
            <h2 style="font-size: 22px; margin: 0 0 8px;">A humanidade foi extinta</h2>
            <p style="color: #aaa; font-size: 14px; margin: 0 0 6px;">
                Ciclo <strong style="color:#fff">#${cycleCount}</strong> encerrado
            </p>
            <p style="color: #aaa; font-size: 14px; margin: 0 0 28px;">
                ${savedGenerationWeights
            ? 'O conhecimento desta geração foi preservado.<br>A próxima começará mais sábia.'
            : 'Nenhum conhecimento foi acumulado ainda.'}
            </p>
            <p style="color:#666; font-size:12px; margin-top: 8px;">${lastSavedText}</p>
            <button id="btn-next-cycle" style="
                background: #4f8ef7; color: white; border: none; border-radius: 10px;
                padding: 12px 32px; font-size: 15px; cursor: pointer; margin-right: 10px;
            ">🔄 Próximo Ciclo</button>
            <button id="btn-reset" style="
                background: transparent; color: #888; border: 1px solid #555;
                border-radius: 10px; padding: 12px 24px; font-size: 15px; cursor: pointer;
            ">Resetar tudo</button>
        </div>
    `;

    document.body.appendChild(modal);

    document.getElementById('btn-next-cycle').addEventListener('click', () => {
        modal.remove();
        extinctionModalActive = false;
        spawnInitialHumans();
        simRunning = true;
        requestRender();
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        modal.remove();
        extinctionModalActive = false;
        savedGenerationWeights = null;
        cycleCount = 0;
        regenerateWorld();
    });
}

self.Build = async () => {
    EventsListener();
    resizeCanvas();
    updateMobileUIState();
    renderResourceGrid();
    await loadKnowledgeFromStorage();
    regenerateWorld();
    startSimulationLoop();
}

self.Build();