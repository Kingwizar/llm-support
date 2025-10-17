package com.example.npone_llm.ui.theme

import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightLuxColors = lightColorScheme(
    primary = GoldYellow,              // boutons, accents
    onPrimary = Color.White,           // texte sur bouton doré
    secondary = SoftBrown,             // teinte secondaire douce
    onSecondary = DeepOlive,
    tertiary = DeepOlive,              // éléments contrastés
    background = IvoryWhite,           // fond principal clair
    onBackground = TextDark,           // texte foncé
    surface = Color.White,             // surfaces (cards, champs)
    onSurface = DeepOlive,             // texte sur surface
    outline = SoftBrown
)

private val DarkLuxColors = darkColorScheme(
    primary = GoldYellow,
    onPrimary = Color.Black,
    secondary = SoftBrown,
    onSecondary = Color.White,
    background = DeepOlive,
    onBackground = Color.White,
    surface = Color(0xFF2B2B25),
    onSurface = Color.White
)

@Composable
fun Npone_llmTheme(
    darkTheme: Boolean = false, // ☀️ par défaut clair et lumineux
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkLuxColors else LightLuxColors

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
