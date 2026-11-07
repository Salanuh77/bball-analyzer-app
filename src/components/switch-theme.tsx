import { Moon, Sun } from 'lucide-react-native';
import { createContext, useState } from "react";
import { Switch } from "react-native";
import { View } from "react-native-reanimated/lib/typescript/Animated";

interface Theme {
    theme: 'dark' | 'light';
}


export const ThemeContext = createContext(true);

const SwitchTheme = () => {
    const [isLight, setisLight] = useState(true);
    const [theme, setTheme] = useState<Theme>({ theme: 'light' });

    return (
        <View>
            <ThemeContext.Provider value={(isLight)}>
                {isLight ? <Sun size={48} /> : <Moon size={48} />}
                <Switch
                    trackColor={{ false: '#767577', true: '#81b0ff' }}
                    onValueChange={() => { isLight ? (setTheme({ theme: 'dark' }), setisLight(false)) : (setTheme({ theme: 'light' }), setisLight(true)) }}
                    value={isLight}
                />
            </ThemeContext.Provider>
        </View>
    )
}