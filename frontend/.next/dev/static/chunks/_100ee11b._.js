(globalThis.TURBOPACK || (globalThis.TURBOPACK = [])).push([typeof document === "object" ? document.currentScript : undefined,
"[project]/lib/translations.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "translations",
    ()=>translations
]);
const en = {
    // ── Nav / Home ──────────────────────────────────────────────
    navVerify: "Verification demo",
    navDashboard: "Dashboard",
    heroBadge: "Video verification platform",
    heroTitle: "Real-time deepfake protection",
    heroDesc: "Connect verus/id to your service and get instant user verification with an ensemble of machine learning models.",
    heroCta: "Try verification",
    heroDashboard: "Open dashboard",
    feat1Title: "5-second recording",
    feat1Desc: "The user records a short video — we check it for deepfake signs in 4–6 seconds.",
    feat2Title: "Model ensemble",
    feat2Desc: "EfficientNet-B7, Vision Transformer, Liveness CNN and more. You choose the models in settings.",
    feat3Title: "Attack analytics",
    feat3Desc: "Detailed statistics on attack types, sources and trends. All in the company dashboard.",
    statAccuracy: "Accuracy",
    statAvgTime: "Avg. time",
    statChecks: "Checks today",
    statTypes: "Attack types",
    footerBrand: "verus/id · Deepfake Detection Platform",
    footerSec: "TLS 1.3 · SOC 2 Type II",
    // ── Verification flow ────────────────────────────────────────
    vRequestFrom: "Request from",
    vTitle: "Confirm it's really you",
    vDesc: "We'll record a short video (5 seconds), check for deepfake signs, and return you back. Video is not saved — only the verification result.",
    vStep1Title: "Camera & microphone",
    vStep1Sub: "One-time permission request",
    vStep2Title: "5-second recording",
    vStep2Sub: "Turn your head, blink",
    vStep3Title: "E-mail & code",
    vStep3Sub: "Identity confirmation",
    vStartFlow: "Start verification",
    vProtected: "Protected by TLS · Session #",
    vCamTitle: "Camera access required",
    vCamGranted: "Access granted",
    vCamDesc: "The browser will request permission. Video is processed on our servers and deleted after verification.",
    vCamReadyDesc: "Ready to record",
    vAllow: "Allow access",
    vContinue: "Continue",
    vIdle: "Place your face in the oval and press 'Start'",
    vCountdown: "Don't move…",
    vLookAt: "Look at the camera",
    vTurnRight: "Turn your head right",
    vBlinkTimes: "Blink several times",
    vRecDone: "Recording complete",
    vStartRec: "Start recording",
    vPreparing: "Preparing…",
    vIsRecording: "Recording…",
    vCancelNote: "YOU CAN CANCEL THE RECORDING AT ANY TIME",
    vEmailTitle: "Enter your e-mail",
    vEmailDesc: "We'll send a one-time confirmation code. The address is used only for the current session.",
    vEmailLabel: "E-mail",
    vHashNote: "Video uploaded. Hash:",
    vSendCode: "Send code",
    vCodeTitle: "Enter the code",
    vCodeSentTo: "Sent to",
    vCodeHint: "Hint:",
    vNoCode: "Didn't receive the code?",
    vResend: "Resend",
    vConfirm: "Confirm",
    vAnalyzingTitle: "Analyzing video",
    vAnalyzingDesc: "Running through selected models. Usually takes 4–6 seconds.",
    vStages: [
        "Loading frames",
        "EfficientNet-B7",
        "Vision Transformer",
        "Liveness CNN",
        "Aggregating results"
    ],
    vPassTitle: "Verification complete",
    vPassDesc: "No deepfake signs detected. Returning you to the site.",
    vPassConfidence: "Confidence",
    vPassLiveness: "Liveness score",
    vPassTime: "Analysis time",
    vPassModels: "Models",
    vReturnTo: "Return to",
    vAutoReturn: "Auto-redirect in 4s…",
    vRetryTitle: "Please try again",
    vRetryDesc: "Systems detected attack signs. This can happen with poor lighting, use of a photo or screen.",
    vAnom: "Spatial-anomaly",
    vLiveness: "Liveness score",
    vLipSync: "Lip-sync drift",
    vHigh: "high",
    vModerate: "moderate",
    vRetryBtn: "Retry recording",
    vCancelReturn: "Cancel and return",
    // ── Dashboard sidebar ────────────────────────────────────────
    dBalance: "Balance",
    dDaysLeft: "≈ 14 days of operation",
    dLastWeek: "Last 7 days",
    dLogout: "Log out",
    dPaneOverview: "Overview",
    dPaneOverviewSub: "Weekly verification summary",
    dPaneAttacks: "Attacks",
    dPaneAttacksSub: "Detailed analytics by attack vectors",
    dPaneModels: "Models",
    dPaneModelsSub: "Ensemble configuration and pricing",
    dPaneBilling: "Billing",
    dPaneBillingSub: "Top-up and charge history",
    dPaneIntegrations: "Integration",
    dPaneIntegrationsSub: "API keys, webhook, redirect URL",
    // ── Overview pane ────────────────────────────────────────────
    dTotalChecks: "TOTAL CHECKS",
    dTotalChecksSub: "+18.2% vs prev. period",
    dAttacksFound: "ATTACKS DETECTED",
    dAttacksFoundSub: "2.66% of traffic",
    dAvgTime: "AVG TIME",
    dAvgTimeSub: "−0.3s vs prev. period",
    dAccuracy: "ACCURACY",
    dAccuracySub: "avg across 3 models",
    dCheckFlow: "Verification flow",
    dCheckFlowSub: "Last week · Σ 1,065",
    dAttacksLabel: "Attacks",
    dCleanLabel: "Clean",
    dAttackTypes: "Attack types",
    dRecentChecks: "Recent verifications",
    dAllBtn: "All →",
    dColId: "ID",
    dColTime: "Time",
    dColResult: "Result",
    dColModels: "Models",
    dColConf: "Confidence",
    dColSource: "Source",
    // ── Models pane ──────────────────────────────────────────────
    dModelsEnabled: "MODELS ENABLED",
    dCostPerCheck: "COST PER CHECK",
    dEnsembleAcc: "ENSEMBLE ACCURACY",
    dEnsembleAccSub: "avg across models",
    dExpLatency: "EXPECTED LATENCY",
    dExpLatencySub: "sequential",
    dModelCatalog: "Model catalog",
    dModelCatalogSub: "Choose models for the ensemble. Cost and latency are summed.",
    dRec: "rec",
    dColAccuracy: "Accuracy",
    dColLatency: "Latency",
    dColPrice: "Price",
    dExtraSettings: "Additional settings",
    dExtraSettingsSub: "Affect cost and accuracy.",
    dExtra0Label: "CPU pre-check before GPU",
    dExtra0Desc: "Filters obviously clean videos on a cheap model",
    dExtra1Label: "Parallel model execution",
    dExtra1Desc: "Reduces latency, doubles GPU cost",
    dExtra2Label: "Save video for audit",
    dExtra2Desc: "Storage 30 days. +₸0.04/check",
    dExtra3Label: "Webhook on each result",
    dExtra3Desc: "POST to your endpoint",
    // ── Attacks pane ─────────────────────────────────────────────
    dVsPrev: "VS. PREVIOUS MONTH",
    dVsPrevVal: "+34.2%",
    dVsPrevSub: "attack attempt growth",
    dMainVector: "MAIN VECTOR",
    dMainVectorVal: "Face swap",
    dMainVectorSub: "142 cases · 41.5%",
    dPeakTime: "PEAK TIME",
    dPeakTimeVal: "21:00–01:00",
    dPeakTimeSub: "64% of traffic",
    dAttackDynTitle: "Attack dynamics",
    dAttackDynSub: "30 days · all types",
    dAllAttempts: "All attempts",
    dBlockedLabel: "Blocked",
    dByType: "By attack type",
    dTopSources: "Top sources",
    dChecksUnit: "checks",
    dAttacksPct: "% attacks",
    dFaceSwap: "Face swap",
    dLipSyncType: "Lip-sync mismatch",
    dReplayType: "Replay (photo/screen)",
    dMaskType: "Mask attack",
    dFullSynth: "Full synthesis",
    // ── Billing pane ─────────────────────────────────────────────
    dCurrentBalance: "CURRENT BALANCE",
    dBalanceSub: "≈ 14 days at current flow (~915 checks/day)",
    dMonthlySpend: "Monthly spend",
    dTopUpTitle: "Top up balance",
    dTopUpDesc: "Instant crediting. +5% bonus from ₸100,000.",
    dTopUpBtn: "Top up for",
    dPayMethods: "VISA · MASTERCARD · MIR · QR",
    dHistoryTitle: "Charge history",
    dHistoryRows: [
        [
            "Oct 28",
            "Charge · 915 checks",
            "−₸ 12,810"
        ],
        [
            "Oct 27",
            "Charge · 1,024 checks",
            "−₸ 14,336"
        ],
        [
            "Oct 26",
            "Top-up · QR",
            "+₸ 100,000"
        ],
        [
            "Oct 26",
            "Charge · 882 checks",
            "−₸ 12,348"
        ],
        [
            "Oct 25",
            "Charge · 904 checks",
            "−₸ 12,656"
        ]
    ],
    // ── Integrations pane ────────────────────────────────────────
    dApiKey: "API key",
    dRedirectUrl: "Redirect URL after verification",
    dWebhook: "Webhook endpoint",
    dSave: "Save",
    dCopy: "Copy"
};
const kz = {
    navVerify: "Тексеру демосы",
    navDashboard: "Жеке кабинет",
    heroBadge: "Бейне-верификация платформасы",
    heroTitle: "Нақты уақытта дипфейктен қорғау",
    heroDesc: "verus/id-ды сервисіңізге қосып, машиналық оқыту модельдерінің жиынтығымен лезде пайдаланушы верификациясын алыңыз.",
    heroCta: "Верификацияны сынау",
    heroDashboard: "Дашбордты ашу",
    feat1Title: "5 секундтық жазба",
    feat1Desc: "Пайдаланушы қысқа бейне жазады — біз оны 4–6 секунда ішінде дипфейк белгілеріне тексереміз.",
    feat2Title: "Модельдер жиынтығы",
    feat2Desc: "EfficientNet-B7, Vision Transformer, Liveness CNN және т.б. Модельдерді параметрлерде өзіңіз таңдайсыз.",
    feat3Title: "Шабуыл аналитикасы",
    feat3Desc: "Шабуыл түрлері, көздері мен динамикасы бойынша егжей-тегжейлі статистика. Бәрі компания дашбордында.",
    statAccuracy: "Дәлдігі",
    statAvgTime: "Орташа уақыт",
    statChecks: "Бүгінгі тексерулер",
    statTypes: "Шабуыл түрлері",
    footerBrand: "verus/id · Дипфейкті анықтау платформасы",
    footerSec: "TLS 1.3 · SOC 2 Type II",
    vRequestFrom: "Сұраным",
    vTitle: "Өзіңіз екеніңізді растаңыз",
    vDesc: "Қысқа бейне жазып (5 секунд), дипфейк белгілеріне тексереміз де сізді кері жібереміз. Бейне сақталмайды — тек тексеру нәтижесі.",
    vStep1Title: "Камера және микрофон",
    vStep1Sub: "Бір рет рұқсат сұрайды",
    vStep2Title: "5 секундтық жазба",
    vStep2Sub: "Басыңызды бұрып, жыпылықтаңыз",
    vStep3Title: "E-mail және код",
    vStep3Sub: "Жеке басты растау",
    vStartFlow: "Тексеруді бастау",
    vProtected: "TLS қорғауы · Сессия #",
    vCamTitle: "Камераға рұқсат қажет",
    vCamGranted: "Рұқсат берілді",
    vCamDesc: "Браузер рұқсат сұрайды. Бейне серверлерімізде өңделіп, тексеруден кейін жойылады.",
    vCamReadyDesc: "Жазуға дайын",
    vAllow: "Рұқсат беру",
    vContinue: "Жалғастыру",
    vIdle: "Бетіңізді сопаға орналастырып «Бастау» батырмасын басыңыз",
    vCountdown: "Қозғалмаңыз…",
    vLookAt: "Камераға қараңыз",
    vTurnRight: "Басыңызды оңға бұрыңыз",
    vBlinkTimes: "Бірнеше рет жыпылықтаңыз",
    vRecDone: "Жазба аяқталды",
    vStartRec: "Жазуды бастау",
    vPreparing: "Дайындалуда…",
    vIsRecording: "Жазылуда…",
    vCancelNote: "ЖАЗБАНЫ КЕЗ КЕЛГЕН УАҚЫТТА ТОҚТАТА АЛАСЫЗ",
    vEmailTitle: "E-mail енгізіңіз",
    vEmailDesc: "Бір реттік растау коды жіберіледі. Мекенжай тек ағымдағы сессия үшін пайдаланылады.",
    vEmailLabel: "E-mail",
    vHashNote: "Бейне жүктелді. Хэш:",
    vSendCode: "Код жіберу",
    vCodeTitle: "Кодты енгізіңіз",
    vCodeSentTo: "Жіберілді",
    vCodeHint: "Кеңес:",
    vNoCode: "Код келмеді ме?",
    vResend: "Қайта жіберу",
    vConfirm: "Растау",
    vAnalyzingTitle: "Бейне талданып жатыр",
    vAnalyzingDesc: "Таңдалған модельдерден өткізіліп жатыр. Әдетте 4–6 секунд кетеді.",
    vStages: [
        "Кадрларды жүктеу",
        "EfficientNet-B7",
        "Vision Transformer",
        "Liveness CNN",
        "Нәтижелерді жинақтау"
    ],
    vPassTitle: "Тексеру аяқталды",
    vPassDesc: "Дипфейк белгілері анықталмады. Сізді сайтқа қайтарамыз.",
    vPassConfidence: "Сенімділік",
    vPassLiveness: "Liveness score",
    vPassTime: "Талдау уақыты",
    vPassModels: "Модельдер",
    vReturnTo: "Оралу",
    vAutoReturn: "4 секундтан кейін автоматты өту…",
    vRetryTitle: "Қайталап көріңіз",
    vRetryDesc: "Жүйелер шабуыл белгілерін анықтады. Бұл нашар жарықта, фото немесе экран пайдаланған кезде болуы мүмкін.",
    vAnom: "Кеңістіктік-аномалия",
    vLiveness: "Liveness score",
    vLipSync: "Ерін синхроны",
    vHigh: "жоғары",
    vModerate: "орташа",
    vRetryBtn: "Жазбаны қайталау",
    vCancelReturn: "Болдырмау және оралу",
    dBalance: "Баланс",
    dDaysLeft: "≈ 14 күн жұмыс",
    dLastWeek: "Соңғы 7 күн",
    dLogout: "Шығу",
    dPaneOverview: "Шолу",
    dPaneOverviewSub: "Аптадағы тексерулер жиынтығы",
    dPaneAttacks: "Шабуылдар",
    dPaneAttacksSub: "Векторлар бойынша егжей-тегжейлі аналитика",
    dPaneModels: "Модельдер",
    dPaneModelsSub: "Жиынтық конфигурациясы және тарифтер",
    dPaneBilling: "Баланс",
    dPaneBillingSub: "Толықтыру және шығын тарихы",
    dPaneIntegrations: "Интеграция",
    dPaneIntegrationsSub: "API кілттері, webhook, redirect URL",
    dTotalChecks: "БАРЛЫҚ ТЕКСЕРУЛЕР",
    dTotalChecksSub: "+18.2% алдыңғы кезеңге",
    dAttacksFound: "ШАБУЫЛДАР АНЫҚТАЛДЫ",
    dAttacksFoundSub: "трафиктің 2.66%",
    dAvgTime: "ОРТАША УАҚЫТ",
    dAvgTimeSub: "алдыңғы кезеңге −0.3с",
    dAccuracy: "ДӘЛДІГІ",
    dAccuracySub: "3 модель бойынша орташа",
    dCheckFlow: "Тексеру ағыны",
    dCheckFlowSub: "Өткен апта · Σ 1,065",
    dAttacksLabel: "Шабуылдар",
    dCleanLabel: "Таза",
    dAttackTypes: "Шабуыл түрлері",
    dRecentChecks: "Соңғы тексерулер",
    dAllBtn: "Барлығы →",
    dColId: "ID",
    dColTime: "Уақыт",
    dColResult: "Нәтиже",
    dColModels: "Модельдер",
    dColConf: "Сенімділік",
    dColSource: "Дереккөз",
    dModelsEnabled: "МОДЕЛЬДЕР ҚОСУЛЫ",
    dCostPerCheck: "ТЕКСЕРУ БАҒАСЫ",
    dEnsembleAcc: "ЖИЫНТЫҚ ДӘЛДІГІ",
    dEnsembleAccSub: "модельдер бойынша орташа",
    dExpLatency: "КҮТІЛЕТІН КІДІРІС",
    dExpLatencySub: "дәйекті",
    dModelCatalog: "Модельдер каталогы",
    dModelCatalogSub: "Жиынтық үшін модельдер таңдаңыз. Баға мен кідіріс жинақталады.",
    dRec: "ұсын",
    dColAccuracy: "Дәлдігі",
    dColLatency: "Кідіріс",
    dColPrice: "Баға",
    dExtraSettings: "Қосымша баптаулар",
    dExtraSettingsSub: "Баға мен дәлдікке әсер етеді.",
    dExtra0Label: "GPU алдында CPU тексеруі",
    dExtra0Desc: "Арзан модельде айқын таза бейнелерді сүзеді",
    dExtra1Label: "Модельдерді параллель іске қосу",
    dExtra1Desc: "Кідірісті азайтады, GPU шығынын екі есе арттырады",
    dExtra2Label: "Аудит үшін бейнені сақтау",
    dExtra2Desc: "30 күн сақтау. +₸0.04/тексеру",
    dExtra3Label: "Әр нәтижеге webhook",
    dExtra3Desc: "Endpoint-іңізге POST",
    dVsPrev: "АЛДЫҢҒЫ АЙМЕН САЛЫСТЫРУ",
    dVsPrevVal: "+34.2%",
    dVsPrevSub: "шабуыл әрекеттерінің өсуі",
    dMainVector: "БАСТЫ ВЕКТОР",
    dMainVectorVal: "Бет ауыстыру",
    dMainVectorSub: "142 жағдай · 41.5%",
    dPeakTime: "ШЫҢ УАҚЫТЫ",
    dPeakTimeVal: "21:00–01:00",
    dPeakTimeSub: "трафиктің 64%",
    dAttackDynTitle: "Шабуыл динамикасы",
    dAttackDynSub: "30 күн · барлық түрлер",
    dAllAttempts: "Барлық әрекеттер",
    dBlockedLabel: "Бұғатталғандар",
    dByType: "Шабуыл түрлері бойынша",
    dTopSources: "Үздік дереккөздер",
    dChecksUnit: "тексеру",
    dAttacksPct: "% шабуыл",
    dFaceSwap: "Бет ауыстыру",
    dLipSyncType: "Ерін синхроны сәйкессіздігі",
    dReplayType: "Қайта ойнату (фото/экран)",
    dMaskType: "Маска шабуылы",
    dFullSynth: "Толық синтез",
    dCurrentBalance: "АҒЫМДАҒЫ БАЛАНС",
    dBalanceSub: "≈ 14 күн ағымдағы ағынмен (~915 тексеру/күн)",
    dMonthlySpend: "Айлық шығын",
    dTopUpTitle: "Балансты толықтыру",
    dTopUpDesc: "Лезде есептеледі. ₸100,000-нан +5% бонус.",
    dTopUpBtn: "Толықтыру сомасы",
    dPayMethods: "VISA · MASTERCARD · MIR · QR",
    dHistoryTitle: "Шығын тарихы",
    dHistoryRows: [
        [
            "28 қаз",
            "Шығын · 915 тексеру",
            "−₸ 12,810"
        ],
        [
            "27 қаз",
            "Шығын · 1,024 тексеру",
            "−₸ 14,336"
        ],
        [
            "26 қаз",
            "Толықтыру · QR",
            "+₸ 100,000"
        ],
        [
            "26 қаз",
            "Шығын · 882 тексеру",
            "−₸ 12,348"
        ],
        [
            "25 қаз",
            "Шығын · 904 тексеру",
            "−₸ 12,656"
        ]
    ],
    dApiKey: "API кілті",
    dRedirectUrl: "Тексеруден кейінгі Redirect URL",
    dWebhook: "Webhook endpoint",
    dSave: "Сақтау",
    dCopy: "Көшіру"
};
const ru = {
    navVerify: "Демо верификации",
    navDashboard: "Личный кабинет",
    heroBadge: "Платформа видео-верификации",
    heroTitle: "Защита от дипфейков в реальном времени",
    heroDesc: "Подключите verus/id к своему сервису и получайте мгновенную верификацию пользователей с ансамблем моделей машинного обучения.",
    heroCta: "Попробовать верификацию",
    heroDashboard: "Открыть дашборд",
    feat1Title: "Запись 5 секунд",
    feat1Desc: "Пользователь записывает короткое видео — мы проверяем его на признаки дипфейка за 4–6 секунд.",
    feat2Title: "Ансамбль моделей",
    feat2Desc: "EfficientNet-B7, Vision Transformer, Liveness CNN и другие. Вы выбираете нужные модели в настройках.",
    feat3Title: "Аналитика атак",
    feat3Desc: "Детальная статистика по типам атак, источникам и динамике. Всё в личном кабинете компании.",
    statAccuracy: "Точность",
    statAvgTime: "Среднее время",
    statChecks: "Проверок сегодня",
    statTypes: "Типов атак",
    footerBrand: "verus/id · Deepfake Detection Platform",
    footerSec: "TLS 1.3 · SOC 2 Type II",
    vRequestFrom: "Запрос от",
    vTitle: "Подтвердите, что вы — это вы",
    vDesc: "Запишем короткое видео (5 секунд), проверим на признаки дипфейка и вернём вас обратно. Видео не сохраняется — только результат проверки.",
    vStep1Title: "Камера и микрофон",
    vStep1Sub: "Запросим доступ один раз",
    vStep2Title: "Запись 5 секунд",
    vStep2Sub: "Поверните голову, моргните",
    vStep3Title: "E-mail и код",
    vStep3Sub: "Подтверждение личности",
    vStartFlow: "Начать проверку",
    vProtected: "Защищено TLS · Сессия #",
    vCamTitle: "Требуется доступ к камере",
    vCamGranted: "Доступ получен",
    vCamDesc: "Браузер запросит разрешение. Видео обрабатывается на наших серверах и удаляется после проверки.",
    vCamReadyDesc: "Можно перейти к записи",
    vAllow: "Разрешить доступ",
    vContinue: "Продолжить",
    vIdle: "Поместите лицо в овал и нажмите «Начать»",
    vCountdown: "Не двигайтесь…",
    vLookAt: "Смотрите в камеру",
    vTurnRight: "Поверните голову вправо",
    vBlinkTimes: "Моргните несколько раз",
    vRecDone: "Запись завершена",
    vStartRec: "Начать запись",
    vPreparing: "Подготовка…",
    vIsRecording: "Идёт запись…",
    vCancelNote: "ВЫ МОЖЕТЕ ОТМЕНИТЬ ЗАПИСЬ В ЛЮБОЙ МОМЕНТ",
    vEmailTitle: "Укажите e-mail",
    vEmailDesc: "Отправим одноразовый код подтверждения. Адрес используется только для текущей сессии.",
    vEmailLabel: "E-mail",
    vHashNote: "Видео уже загружено. Хеш:",
    vSendCode: "Отправить код",
    vCodeTitle: "Введите код",
    vCodeSentTo: "Отправили на",
    vCodeHint: "Подсказка:",
    vNoCode: "Не пришёл код?",
    vResend: "Отправить снова",
    vConfirm: "Подтвердить",
    vAnalyzingTitle: "Анализируем видео",
    vAnalyzingDesc: "Прогоняем через выбранные модели. Обычно занимает 4–6 секунд.",
    vStages: [
        "Загрузка кадров",
        "EfficientNet-B7",
        "Vision Transformer",
        "Liveness CNN",
        "Агрегация результатов"
    ],
    vPassTitle: "Проверка завершена",
    vPassDesc: "Подозрений на дипфейк не выявлено. Возвращаем вас на сайт.",
    vPassConfidence: "Уверенность",
    vPassLiveness: "Liveness score",
    vPassTime: "Время анализа",
    vPassModels: "Моделей",
    vReturnTo: "Вернуться на",
    vAutoReturn: "Автопереход через 4с…",
    vRetryTitle: "Попробуйте снова",
    vRetryDesc: "Системы выявили признаки атаки. Это может произойти при плохом освещении, использовании фото или экрана.",
    vAnom: "Spatial-anomaly",
    vLiveness: "Liveness score",
    vLipSync: "Lip-sync drift",
    vHigh: "высокая",
    vModerate: "умеренный",
    vRetryBtn: "Повторить запись",
    vCancelReturn: "Отменить и вернуться",
    dBalance: "Баланс",
    dDaysLeft: "≈ 14 дней работы",
    dLastWeek: "Последние 7 дней",
    dLogout: "Выйти",
    dPaneOverview: "Обзор",
    dPaneOverviewSub: "Сводка по проверкам за неделю",
    dPaneAttacks: "Атаки",
    dPaneAttacksSub: "Детальная аналитика по векторам",
    dPaneModels: "Модели",
    dPaneModelsSub: "Конфигурация ансамбля и тарифов",
    dPaneBilling: "Баланс",
    dPaneBillingSub: "Пополнение и история списаний",
    dPaneIntegrations: "Интеграция",
    dPaneIntegrationsSub: "API ключи, webhook, redirect URL",
    dTotalChecks: "ВСЕГО ПРОВЕРОК",
    dTotalChecksSub: "+18.2% к пред. периоду",
    dAttacksFound: "ВЫЯВЛЕНО АТАК",
    dAttacksFoundSub: "2.66% от потока",
    dAvgTime: "СРЕДНЕЕ ВРЕМЯ",
    dAvgTimeSub: "−0.3s к пред. периоду",
    dAccuracy: "ТОЧНОСТЬ",
    dAccuracySub: "avg по 3 моделям",
    dCheckFlow: "Поток проверок",
    dCheckFlowSub: "За последнюю неделю · Σ 1,065",
    dAttacksLabel: "Атаки",
    dCleanLabel: "Чистые",
    dAttackTypes: "Типы атак",
    dRecentChecks: "Последние проверки",
    dAllBtn: "Все →",
    dColId: "ID",
    dColTime: "Время",
    dColResult: "Результат",
    dColModels: "Модели",
    dColConf: "Уверенность",
    dColSource: "Источник",
    dModelsEnabled: "МОДЕЛЕЙ ВКЛЮЧЕНО",
    dCostPerCheck: "СТОИМОСТЬ ЗА ПРОВЕРКУ",
    dEnsembleAcc: "АНСАМБЛЬНАЯ ТОЧНОСТЬ",
    dEnsembleAccSub: "среднее по моделям",
    dExpLatency: "ОЖИДАЕМАЯ ЗАДЕРЖКА",
    dExpLatencySub: "последовательно",
    dModelCatalog: "Каталог моделей",
    dModelCatalogSub: "Выберите модели для ансамбля. Цена и задержка суммируются.",
    dRec: "rec",
    dColAccuracy: "Точность",
    dColLatency: "Задержка",
    dColPrice: "Цена",
    dExtraSettings: "Доп. настройки",
    dExtraSettingsSub: "Влияют на стоимость и точность.",
    dExtra0Label: "CPU-проверка перед GPU",
    dExtra0Desc: "Отсеивает заведомо чистые видео на дешёвой модели",
    dExtra1Label: "Параллельный запуск моделей",
    dExtra1Desc: "Уменьшает задержку, удваивает стоимость GPU",
    dExtra2Label: "Сохранять видео для аудита",
    dExtra2Desc: "Хранение 30 дней. +₸0.04/проверка",
    dExtra3Label: "Webhook на каждый результат",
    dExtra3Desc: "POST на ваш endpoint",
    dVsPrev: "VS. ПРЕДЫДУЩИЙ МЕСЯЦ",
    dVsPrevVal: "+34.2%",
    dVsPrevSub: "рост попыток атак",
    dMainVector: "ОСНОВНОЙ ВЕКТОР",
    dMainVectorVal: "Face swap",
    dMainVectorSub: "142 случая · 41.5%",
    dPeakTime: "ПИКОВОЕ ВРЕМЯ",
    dPeakTimeVal: "21:00–01:00",
    dPeakTimeSub: "64% от потока",
    dAttackDynTitle: "Динамика атак",
    dAttackDynSub: "30 дней · все типы",
    dAllAttempts: "Все попытки",
    dBlockedLabel: "Успешные блокировки",
    dByType: "По типам атак",
    dTopSources: "Топ источников",
    dChecksUnit: "проверок",
    dAttacksPct: "% атак",
    dFaceSwap: "Face swap",
    dLipSyncType: "Lip-sync mismatch",
    dReplayType: "Replay (фото/экран)",
    dMaskType: "Mask attack",
    dFullSynth: "Полный синтез",
    dCurrentBalance: "ТЕКУЩИЙ БАЛАНС",
    dBalanceSub: "≈ 14 дней при текущем потоке (~915 проверок/день)",
    dMonthlySpend: "Расход за месяц",
    dTopUpTitle: "Пополнить баланс",
    dTopUpDesc: "Зачисление мгновенно. Бонус +5% от ₸100,000.",
    dTopUpBtn: "Пополнить на",
    dPayMethods: "ВИЗА · MASTERCARD · МИР · СБП",
    dHistoryTitle: "История списаний",
    dHistoryRows: [
        [
            "28 окт",
            "Списание · 915 проверок",
            "−₸ 12,810"
        ],
        [
            "27 окт",
            "Списание · 1,024 проверок",
            "−₸ 14,336"
        ],
        [
            "26 окт",
            "Пополнение · СБП",
            "+₸ 100,000"
        ],
        [
            "26 окт",
            "Списание · 882 проверок",
            "−₸ 12,348"
        ],
        [
            "25 окт",
            "Списание · 904 проверок",
            "−₸ 12,656"
        ]
    ],
    dApiKey: "API ключ",
    dRedirectUrl: "Redirect URL после проверки",
    dWebhook: "Webhook endpoint",
    dSave: "Сохранить",
    dCopy: "Копировать"
};
const translations = {
    en,
    ru,
    kz
};
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/components/i18n-provider.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "I18nProvider",
    ()=>I18nProvider,
    "LangSwitcher",
    ()=>LangSwitcher,
    "useI18n",
    ()=>useI18n
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$translations$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/translations.ts [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature(), _s1 = __turbopack_context__.k.signature(), _s2 = __turbopack_context__.k.signature();
"use client";
;
;
const I18nContext = /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["createContext"])({
    lang: "en",
    setLang: ()=>{},
    t: __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$translations$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["translations"].en
});
function I18nProvider({ children }) {
    _s();
    const [lang, setLang] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])("en");
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(I18nContext.Provider, {
        value: {
            lang,
            setLang,
            t: __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$translations$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["translations"][lang]
        },
        children: children
    }, void 0, false, {
        fileName: "[project]/components/i18n-provider.tsx",
        lineNumber: 21,
        columnNumber: 5
    }, this);
}
_s(I18nProvider, "zBAk1dGmlP9YgPBD/UW9ro1+pX8=");
_c = I18nProvider;
function useI18n() {
    _s1();
    return (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useContext"])(I18nContext);
}
_s1(useI18n, "gDsCjeeItUuvgOWf1v4qoK9RF6k=");
function LangSwitcher() {
    _s2();
    const { lang, setLang } = useI18n();
    const opts = [
        {
            value: "en",
            label: "EN"
        },
        {
            value: "kz",
            label: "ҚАЗ"
        }
    ];
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        style: {
            display: "flex",
            borderRadius: 8,
            border: "1px solid var(--line-2)",
            overflow: "hidden",
            background: "var(--surface)"
        },
        children: opts.map((o)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                onClick: ()=>setLang(o.value),
                style: {
                    appearance: "none",
                    border: 0,
                    cursor: "pointer",
                    padding: "6px 12px",
                    fontSize: 12,
                    fontWeight: 500,
                    fontFamily: "var(--font-mono)",
                    letterSpacing: "0.03em",
                    background: lang === o.value ? "var(--ink)" : "transparent",
                    color: lang === o.value ? "#fff" : "var(--muted)",
                    transition: "background 0.12s, color 0.12s"
                },
                children: o.label
            }, o.value, false, {
                fileName: "[project]/components/i18n-provider.tsx",
                lineNumber: 49,
                columnNumber: 9
            }, this))
    }, void 0, false, {
        fileName: "[project]/components/i18n-provider.tsx",
        lineNumber: 39,
        columnNumber: 5
    }, this);
}
_s2(LangSwitcher, "J8Oq2vyHTuijWNHrr0NcMHDUVVg=", false, function() {
    return [
        useI18n
    ];
});
_c1 = LangSwitcher;
var _c, _c1;
__turbopack_context__.k.register(_c, "I18nProvider");
__turbopack_context__.k.register(_c1, "LangSwitcher");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/node_modules/next/dist/compiled/react/cjs/react-jsx-dev-runtime.development.js [app-client] (ecmascript)", ((__turbopack_context__, module, exports) => {
"use strict";

/**
 * @license React
 * react-jsx-dev-runtime.development.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */ var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = /*#__PURE__*/ __turbopack_context__.i("[project]/node_modules/next/dist/build/polyfills/process.js [app-client] (ecmascript)");
"use strict";
"production" !== ("TURBOPACK compile-time value", "development") && function() {
    function getComponentNameFromType(type) {
        if (null == type) return null;
        if ("function" === typeof type) return type.$$typeof === REACT_CLIENT_REFERENCE ? null : type.displayName || type.name || null;
        if ("string" === typeof type) return type;
        switch(type){
            case REACT_FRAGMENT_TYPE:
                return "Fragment";
            case REACT_PROFILER_TYPE:
                return "Profiler";
            case REACT_STRICT_MODE_TYPE:
                return "StrictMode";
            case REACT_SUSPENSE_TYPE:
                return "Suspense";
            case REACT_SUSPENSE_LIST_TYPE:
                return "SuspenseList";
            case REACT_ACTIVITY_TYPE:
                return "Activity";
            case REACT_VIEW_TRANSITION_TYPE:
                return "ViewTransition";
        }
        if ("object" === typeof type) switch("number" === typeof type.tag && console.error("Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."), type.$$typeof){
            case REACT_PORTAL_TYPE:
                return "Portal";
            case REACT_CONTEXT_TYPE:
                return type.displayName || "Context";
            case REACT_CONSUMER_TYPE:
                return (type._context.displayName || "Context") + ".Consumer";
            case REACT_FORWARD_REF_TYPE:
                var innerType = type.render;
                type = type.displayName;
                type || (type = innerType.displayName || innerType.name || "", type = "" !== type ? "ForwardRef(" + type + ")" : "ForwardRef");
                return type;
            case REACT_MEMO_TYPE:
                return innerType = type.displayName || null, null !== innerType ? innerType : getComponentNameFromType(type.type) || "Memo";
            case REACT_LAZY_TYPE:
                innerType = type._payload;
                type = type._init;
                try {
                    return getComponentNameFromType(type(innerType));
                } catch (x) {}
        }
        return null;
    }
    function testStringCoercion(value) {
        return "" + value;
    }
    function checkKeyStringCoercion(value) {
        try {
            testStringCoercion(value);
            var JSCompiler_inline_result = !1;
        } catch (e) {
            JSCompiler_inline_result = !0;
        }
        if (JSCompiler_inline_result) {
            JSCompiler_inline_result = console;
            var JSCompiler_temp_const = JSCompiler_inline_result.error;
            var JSCompiler_inline_result$jscomp$0 = "function" === typeof Symbol && Symbol.toStringTag && value[Symbol.toStringTag] || value.constructor.name || "Object";
            JSCompiler_temp_const.call(JSCompiler_inline_result, "The provided key is an unsupported type %s. This value must be coerced to a string before using it here.", JSCompiler_inline_result$jscomp$0);
            return testStringCoercion(value);
        }
    }
    function getTaskName(type) {
        if (type === REACT_FRAGMENT_TYPE) return "<>";
        if ("object" === typeof type && null !== type && type.$$typeof === REACT_LAZY_TYPE) return "<...>";
        try {
            var name = getComponentNameFromType(type);
            return name ? "<" + name + ">" : "<...>";
        } catch (x) {
            return "<...>";
        }
    }
    function getOwner() {
        var dispatcher = ReactSharedInternals.A;
        return null === dispatcher ? null : dispatcher.getOwner();
    }
    function UnknownOwner() {
        return Error("react-stack-top-frame");
    }
    function hasValidKey(config) {
        if (hasOwnProperty.call(config, "key")) {
            var getter = Object.getOwnPropertyDescriptor(config, "key").get;
            if (getter && getter.isReactWarning) return !1;
        }
        return void 0 !== config.key;
    }
    function defineKeyPropWarningGetter(props, displayName) {
        function warnAboutAccessingKey() {
            specialPropKeyWarningShown || (specialPropKeyWarningShown = !0, console.error("%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://react.dev/link/special-props)", displayName));
        }
        warnAboutAccessingKey.isReactWarning = !0;
        Object.defineProperty(props, "key", {
            get: warnAboutAccessingKey,
            configurable: !0
        });
    }
    function elementRefGetterWithDeprecationWarning() {
        var componentName = getComponentNameFromType(this.type);
        didWarnAboutElementRef[componentName] || (didWarnAboutElementRef[componentName] = !0, console.error("Accessing element.ref was removed in React 19. ref is now a regular prop. It will be removed from the JSX Element type in a future release."));
        componentName = this.props.ref;
        return void 0 !== componentName ? componentName : null;
    }
    function ReactElement(type, key, props, owner, debugStack, debugTask) {
        var refProp = props.ref;
        type = {
            $$typeof: REACT_ELEMENT_TYPE,
            type: type,
            key: key,
            props: props,
            _owner: owner
        };
        null !== (void 0 !== refProp ? refProp : null) ? Object.defineProperty(type, "ref", {
            enumerable: !1,
            get: elementRefGetterWithDeprecationWarning
        }) : Object.defineProperty(type, "ref", {
            enumerable: !1,
            value: null
        });
        type._store = {};
        Object.defineProperty(type._store, "validated", {
            configurable: !1,
            enumerable: !1,
            writable: !0,
            value: 0
        });
        Object.defineProperty(type, "_debugInfo", {
            configurable: !1,
            enumerable: !1,
            writable: !0,
            value: null
        });
        Object.defineProperty(type, "_debugStack", {
            configurable: !1,
            enumerable: !1,
            writable: !0,
            value: debugStack
        });
        Object.defineProperty(type, "_debugTask", {
            configurable: !1,
            enumerable: !1,
            writable: !0,
            value: debugTask
        });
        Object.freeze && (Object.freeze(type.props), Object.freeze(type));
        return type;
    }
    function jsxDEVImpl(type, config, maybeKey, isStaticChildren, debugStack, debugTask) {
        var children = config.children;
        if (void 0 !== children) if (isStaticChildren) if (isArrayImpl(children)) {
            for(isStaticChildren = 0; isStaticChildren < children.length; isStaticChildren++)validateChildKeys(children[isStaticChildren]);
            Object.freeze && Object.freeze(children);
        } else console.error("React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead.");
        else validateChildKeys(children);
        if (hasOwnProperty.call(config, "key")) {
            children = getComponentNameFromType(type);
            var keys = Object.keys(config).filter(function(k) {
                return "key" !== k;
            });
            isStaticChildren = 0 < keys.length ? "{key: someKey, " + keys.join(": ..., ") + ": ...}" : "{key: someKey}";
            didWarnAboutKeySpread[children + isStaticChildren] || (keys = 0 < keys.length ? "{" + keys.join(": ..., ") + ": ...}" : "{}", console.error('A props object containing a "key" prop is being spread into JSX:\n  let props = %s;\n  <%s {...props} />\nReact keys must be passed directly to JSX without using spread:\n  let props = %s;\n  <%s key={someKey} {...props} />', isStaticChildren, children, keys, children), didWarnAboutKeySpread[children + isStaticChildren] = !0);
        }
        children = null;
        void 0 !== maybeKey && (checkKeyStringCoercion(maybeKey), children = "" + maybeKey);
        hasValidKey(config) && (checkKeyStringCoercion(config.key), children = "" + config.key);
        if ("key" in config) {
            maybeKey = {};
            for(var propName in config)"key" !== propName && (maybeKey[propName] = config[propName]);
        } else maybeKey = config;
        children && defineKeyPropWarningGetter(maybeKey, "function" === typeof type ? type.displayName || type.name || "Unknown" : type);
        return ReactElement(type, children, maybeKey, getOwner(), debugStack, debugTask);
    }
    function validateChildKeys(node) {
        isValidElement(node) ? node._store && (node._store.validated = 1) : "object" === typeof node && null !== node && node.$$typeof === REACT_LAZY_TYPE && ("fulfilled" === node._payload.status ? isValidElement(node._payload.value) && node._payload.value._store && (node._payload.value._store.validated = 1) : node._store && (node._store.validated = 1));
    }
    function isValidElement(object) {
        return "object" === typeof object && null !== object && object.$$typeof === REACT_ELEMENT_TYPE;
    }
    var React = __turbopack_context__.r("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)"), REACT_ELEMENT_TYPE = Symbol.for("react.transitional.element"), REACT_PORTAL_TYPE = Symbol.for("react.portal"), REACT_FRAGMENT_TYPE = Symbol.for("react.fragment"), REACT_STRICT_MODE_TYPE = Symbol.for("react.strict_mode"), REACT_PROFILER_TYPE = Symbol.for("react.profiler"), REACT_CONSUMER_TYPE = Symbol.for("react.consumer"), REACT_CONTEXT_TYPE = Symbol.for("react.context"), REACT_FORWARD_REF_TYPE = Symbol.for("react.forward_ref"), REACT_SUSPENSE_TYPE = Symbol.for("react.suspense"), REACT_SUSPENSE_LIST_TYPE = Symbol.for("react.suspense_list"), REACT_MEMO_TYPE = Symbol.for("react.memo"), REACT_LAZY_TYPE = Symbol.for("react.lazy"), REACT_ACTIVITY_TYPE = Symbol.for("react.activity"), REACT_VIEW_TRANSITION_TYPE = Symbol.for("react.view_transition"), REACT_CLIENT_REFERENCE = Symbol.for("react.client.reference"), ReactSharedInternals = React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE, hasOwnProperty = Object.prototype.hasOwnProperty, isArrayImpl = Array.isArray, createTask = console.createTask ? console.createTask : function() {
        return null;
    };
    React = {
        react_stack_bottom_frame: function(callStackForError) {
            return callStackForError();
        }
    };
    var specialPropKeyWarningShown;
    var didWarnAboutElementRef = {};
    var unknownOwnerDebugStack = React.react_stack_bottom_frame.bind(React, UnknownOwner)();
    var unknownOwnerDebugTask = createTask(getTaskName(UnknownOwner));
    var didWarnAboutKeySpread = {};
    exports.Fragment = REACT_FRAGMENT_TYPE;
    exports.jsxDEV = function(type, config, maybeKey, isStaticChildren) {
        var trackActualOwner = 1e4 > ReactSharedInternals.recentlyCreatedOwnerStacks++;
        if (trackActualOwner) {
            var previousStackTraceLimit = Error.stackTraceLimit;
            Error.stackTraceLimit = 10;
            var debugStackDEV = Error("react-stack-top-frame");
            Error.stackTraceLimit = previousStackTraceLimit;
        } else debugStackDEV = unknownOwnerDebugStack;
        return jsxDEVImpl(type, config, maybeKey, isStaticChildren, debugStackDEV, trackActualOwner ? createTask(getTaskName(type)) : unknownOwnerDebugTask);
    };
}();
}),
"[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)", ((__turbopack_context__, module, exports) => {
"use strict";

var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = /*#__PURE__*/ __turbopack_context__.i("[project]/node_modules/next/dist/build/polyfills/process.js [app-client] (ecmascript)");
'use strict';
if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
;
else {
    module.exports = __turbopack_context__.r("[project]/node_modules/next/dist/compiled/react/cjs/react-jsx-dev-runtime.development.js [app-client] (ecmascript)");
}
}),
]);

//# sourceMappingURL=_100ee11b._.js.map