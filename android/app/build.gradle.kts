plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.mit.rlx"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.mit.rlx.demo"
        minSdk = 26
        targetSdk = 35
        // Tracks the workspace version in the root Cargo.toml — the demo bundles
        // librlx_jni.so built from it, so a build.gradle version that drifts
        // makes an installed APK impossible to trace back to a native build.
        // versionCode is derived: major*10000 + minor*100 + patch.
        versionCode = 216
        versionName = "0.2.16"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    sourceSets {
        getByName("androidTest") {
            // Instrumentation runs in com.mit.rlx.demo.test — ship the same
            // .so as the main APK so RlxNative.loadLibrary succeeds.
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.1")
    androidTestImplementation("androidx.test:rules:1.6.1")
}
