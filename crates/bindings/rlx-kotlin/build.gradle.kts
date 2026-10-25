// RLX — versatile ML compiler + runtime. MIT OR Apache-2.0.
//
// Kotlin (JVM) bindings over the generated JNI glue. The glue itself
// (src/main/c) is built with the NDK for Android, or any C compiler for the
// desktop JVM; point -Prlx.jni.lib at the built librlx_jni.
//
//   gradle rlxTest -Prlx.jni.lib=<path>             # src/test: the test program
//   gradle example -Pname=training -Prlx.jni.lib=<path>

plugins {
    kotlin("jvm") version "2.2.20"
}

group = "org.rlx"
version = "0.2.19"

repositories { mavenCentral() }

// Bytecode for Java 17+ (and Android), built with whichever JDK runs Gradle.
java { targetCompatibility = JavaVersion.VERSION_17; sourceCompatibility = JavaVersion.VERSION_17 }
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

sourceSets {
    named("test") { kotlin.srcDir("examples") }
}

val jniLib = providers.gradleProperty("rlx.jni.lib").orNull

fun JavaExec.rlx(main: String) {
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set(main)
    jniLib?.let { systemProperty("rlx.jni.lib", it) }
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}

tasks.register<JavaExec>("rlxTest") {
    group = "verification"
    rlx("org.rlx.kt.RLXTestKt")
}

tasks.register<JavaExec>("example") {
    group = "application"
    val name = providers.gradleProperty("name").getOrElse("inference")
    rlx("org.rlx.kt.examples." + name.replaceFirstChar { it.uppercase() } + "Kt")
}
